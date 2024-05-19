import argparse
import logging
import os

import torch
from torch.export import export
from transformers import AutoModelForSemanticSegmentation, AutoModelForSequenceClassification

from quantized_training.codegen.shape_prop import ShapeProp

logging.basicConfig(level=logging.DEBUG)


def replace_interpolate():
    from torch.library import Library, impl

    template = (
        "interpolate(Tensor input, SymInt[] size, float[]? scale_factor = None,"
        "str mode = 'nearest', bool? align_corners = None, "
        "bool? recompute_scale_factor = None, bool antialias = False) -> Tensor"
    )

    global m
    m = Library("custom", "DEF")
    m.define(template)

    orig_interpolate = torch.nn.functional.interpolate

    @impl(m, "interpolate", "CompositeExplicitAutograd")
    def interpolate(*args, **kwargs):
        return orig_interpolate(*args, **kwargs)

    torch.nn.functional.interpolate = torch.ops.custom.interpolate


def _pair_conv_bn(layers):
    pairs = []
    layer_dict = {}

    # Organize layers by prefix
    for layer in layers:
        prefix = '.'.join(layer.split('.')[:-1])
        if prefix not in layer_dict:
            layer_dict[prefix] = []
        layer_dict[prefix].append(layer)

    # Find pairs of conv and bn
    for prefix, layer_list in layer_dict.items():
        if "downsample" in prefix:
            pairs.append([f"{prefix}.0", f"{prefix}.1"])
        else:
            conv_layers = sorted([l for l in layer_list if 'conv' in l])
            bn_layers = sorted([l for l in layer_list if 'bn' in l])

            # Pair each conv with its corresponding bn
            for conv, bn in zip(conv_layers, bn_layers):
                pairs.append([conv, bn])

    return pairs


def flatten(mixed_list):
    flattened_list = []
    for element in mixed_list:
        if isinstance(element, list):
            flattened_list.extend(flatten(element))
        else:
            flattened_list.append(element)
    return flattened_list


def transform(
        model, example_args, example_kwargs=None, *, print_graph=False,
        generate_graph=False, output_file="compute_graph", output_dir=None):
    if example_kwargs is None:
        example_kwargs = {}

    exported_program: torch.export.ExportedProgram = \
        export(model, example_args, example_kwargs)

    if print_graph:
        print(exported_program)

    shape_prop = ShapeProp(exported_program.module())
    shape_prop.transform()

    if print_graph:
        shape_prop.graph.print_tabular()

    pt_out = model(*example_args, **example_kwargs)
    uplifted_args = flatten(list(example_args)) + list(example_kwargs.values())
    gm_out = shape_prop.propagate(*uplifted_args)[0][0]
    # gm_out = shape_prop.mod(*example_args, *list(example_kwargs.values()))

    params = shape_prop.gen_code(os.path.join(output_dir, "tensor_files"))
    with open(os.path.join(output_dir, 'params.pb'), 'wb') as f:
        f.write(params.SerializeToString())

    import json
    from google.protobuf.json_format import MessageToDict

    data = MessageToDict(params)
    print(json.dumps(data, indent=4))

    if generate_graph:
        shape_prop.gen_compute_graph(os.path.join(output_dir, output_file))

    return (shape_prop, pt_out, gm_out)


if __name__ == "__main__":
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--print_graph", action="store_true")
    parser.add_argument("--generate_graph", action="store_true")
    parser.add_argument("--output_dir", default=None, help="Output directory for generated tensor files")
    args = parser.parse_args()

    if "resnet50" == args.model:
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.eval()
        
        module_names = [name for name, _ in model.named_modules()]
        modules_to_fuse = _pair_conv_bn(module_names)
        model = torch.ao.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

        example_args = (torch.randn(1, 3, 224, 224),)
        _, pt_out, gm_out = transform(
            model,
            example_args,
            print_graph=args.print_graph,
            generate_graph=args.generate_graph,
            output_file="resnet50",
            output_dir=args.output_dir,
        )

    if "segformer" == args.model:
        replace_interpolate()
        model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        model.eval()

        modules_to_fuse = ["decode_head.linear_fuse", "decode_head.batch_norm"]
        model = torch.ao.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

        example_args = (torch.randn(1, 3, 512, 672),)
        _, pt_out, gm_out = transform(
            model,
            example_args,
            print_graph=args.print_graph,
            generate_graph=args.generate_graph,
            output_file="segformer",
            output_dir=args.output_dir,
        )
        pt_out = pt_out.logits

    if "mobilebert" == args.model:
        model = AutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased")
        example_args = (torch.randint(0, 30522, (1, 128), dtype=torch.long),)
        _, pt_out, gm_out = transform(
            model,
            example_args,
            print_graph=args.print_graph,
            generate_graph=args.generate_graph,
            output_file="mobilebert",
            output_dir=args.output_dir,
        )
        pt_out = pt_out.logits

    if "mobilebert_no_embed" == args.model:
        from transformers import MobileBertForSequenceClassification
        model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased")
        input_ids = torch.randint(0, 30522, (1, 128), dtype=torch.long)
        position_ids = torch.ones((1, 128), dtype=torch.long)
        input_shape = input_ids.size()

        attention_mask = torch.ones(input_shape)
        # head_mask = torch.ones(model.config.num_attention_heads)
        head_mask = None
        token_type_ids = torch.zeros(input_shape, dtype=torch.long)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = model.mobilebert.get_extended_attention_mask(attention_mask, input_shape)

        head_mask = model.mobilebert.get_head_mask(head_mask, model.config.num_hidden_layers)

        embedding_output = model.mobilebert.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
        )
        example_args = (embedding_output, extended_attention_mask, head_mask)
        example_kwargs = {"return_dict": False}
        _, pt_out, gm_out = transform(
            model.mobilebert.encoder,
            example_args,
            example_kwargs,
            print_graph=args.print_graph,
            generate_graph=args.generate_graph,
            output_file="mobilebert",
            output_dir=args.output_dir,
        )
        pt_out = pt_out[0]

    try:
        assert torch.all(pt_out == gm_out), "Outputs do not match"
    except:
        print(pt_out)
        print(gm_out)
