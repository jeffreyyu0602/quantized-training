import argparse
import copy
import os

import torch
from datasets import load_dataset
from google.protobuf import text_format
from google.protobuf.json_format import MessageToJson
from torch._export import capture_pre_autograd_graph
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSemanticSegmentation,
    AutoImageProcessor,
)
from tqdm import tqdm

from quantized_training import add_training_args, get_quantizer, prepare_pt2e, convert_pt2e
from quantized_training.codegen.mapping import gen_code, gen_compute_graph, fuse_operator
from quantized_training.quantizer.xnnpack_quantizer_utils import _convert_scalars_to_attrs


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
    model: torch.fx.GraphModule,
    example_args,
    example_kwargs=None,
    *,
    output_file="compute_graph",
    output_dir=None
):
    if example_kwargs is None:
        example_kwargs = {}

    if isinstance(model, torch.fx.GraphModule):
        gm = copy.deepcopy(model)
    else:
        gm = capture_pre_autograd_graph(model, example_args, example_kwargs)
        _convert_scalars_to_attrs(gm)

    gm = fuse_operator(gm)
    gm.graph.print_tabular()

    pt_out = model(*example_args, **example_kwargs)
    gm_out = gm(*example_args, *list(example_kwargs.values()))

    uplifted_args = flatten(list(example_args)) + list(example_kwargs.values())
    params = gen_code(
        gm,
        uplifted_args,
        os.path.join(output_dir, "tensor_files")
    )

    with open(os.path.join(output_dir, 'params.pb'), 'wb') as f:
        f.write(params.SerializeToString())

    with open(os.path.join(output_dir, 'params.txt'), "w") as f:
        f.write(text_format.MessageToString(params))

    with open(os.path.join(output_dir, 'params.json'), "w") as f:
        f.write(MessageToJson(params))

    layers = [p.name for p in params.params]
    with open(os.path.join(output_dir, 'layers.txt'), 'w') as f:
        f.write('\n'.join(layers))

    gen_compute_graph(gm, os.path.join(output_dir, output_file))

    return pt_out, gm_out


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=10)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--output_dir", required=True, help="Output directory for generated tensor files")
    add_training_args(parser)
    args = parser.parse_args()

    if "resnet18" == args.model:
        model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

        module_names = [name for name, _ in model.named_modules()]
        modules_to_fuse = _pair_conv_bn(module_names)
        model = torch.ao.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

        # Accelerator only supports 2x2 maxpool
        model.maxpool.kernel_size = (2, 2)
        model.maxpool.padding = 0

        example_args = (torch.randn(1, 3, 224, 224),)
        pt_out, gm_out = transform(
            model,
            example_args,
            output_file="resnet18",
            output_dir=args.output_dir,
        )

    if "qresnet18" == args.model:
        model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        model.bfloat16()

        module_names = [name for name, _ in model.named_modules()]
        modules_to_fuse = _pair_conv_bn(module_names)
        model = torch.ao.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

        # Accelerator only supports 2x2 maxpool
        model.maxpool.kernel_size = (2, 2)
        model.maxpool.padding = 0

        quantizer = get_quantizer(args.activation, args.weight)
        example_args = (torch.randn(1, 3, 224, 224, dtype=torch.bfloat16),)
        model = prepare_pt2e(model, quantizer, example_args)

        dataset = load_dataset("zh-plus/tiny-imagenet")

        image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

        for i in tqdm(range(10)):
            inputs = image_processor(dataset['train'][i]["image"], return_tensors="pt")
            with torch.no_grad():
                model(inputs.pixel_values.bfloat16())

        convert_pt2e(model)

        pt_out, gm_out = transform(
            model,
            example_args,
            output_file="qresnet18",
            output_dir=args.output_dir,
        )

    if "resnet50" == args.model:
        model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()

        module_names = [name for name, _ in model.named_modules()]
        modules_to_fuse = _pair_conv_bn(module_names)
        model = torch.ao.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

        example_args = (torch.randn(1, 3, 224, 224),)
        pt_out, gm_out = transform(
            model,
            example_args,
            output_file="resnet50",
            output_dir=args.output_dir,
        )

    if "segformer" == args.model:
        replace_interpolate()
        model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").eval()

        modules_to_fuse = ["decode_head.linear_fuse", "decode_head.batch_norm"]
        model = torch.ao.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

        example_args = (torch.randn(1, 3, 512, 672),)
        pt_out, gm_out = transform(
            model,
            example_args,
            output_file="segformer",
            output_dir=args.output_dir,
        )
        pt_out = pt_out.logits

    if "mobilebert" == args.model:
        model = AutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased")
        example_args = (torch.randint(0, 30522, (1, 128), dtype=torch.long),)
        pt_out, gm_out = transform(
            model,
            example_args,
            output_file="mobilebert",
            output_dir=args.output_dir,
        )
        pt_out = pt_out.logits

    if "mobilebert_no_embed" == args.model:
        model = AutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased")

        input_ids = torch.randint(0, 30522, (1, 128), dtype=torch.long)
        input_shape = input_ids.size()
        attention_mask = torch.ones(input_shape)
        token_type_ids = torch.zeros(input_shape, dtype=torch.long)
        position_ids = torch.ones((1, 128), dtype=torch.long)
        head_mask = None

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = model.mobilebert.get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = model.mobilebert.get_head_mask(head_mask, model.config.num_hidden_layers)

        embedding_output = model.mobilebert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        example_args = (embedding_output, extended_attention_mask, head_mask)
        example_kwargs = {"return_dict": False}

        class MobileBertNoEmbed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mobilebert = model.mobilebert
                self.classifier = model.classifier

            def forward(self, *args, **kwargs):
                hidden_states = model.mobilebert.encoder(*args, **kwargs)[0]
                first_token_tensor = hidden_states[:, 0]
                output = model.classifier(first_token_tensor)
                return output

        pt_out, gm_out = transform(
            MobileBertNoEmbed(),
            example_args,
            example_kwargs,
            output_file="mobilebert",
            output_dir=args.output_dir,
        )
        pt_out = pt_out[0]

    try:
        assert torch.all(pt_out == gm_out)
        print("Results match")
    except Exception as e:
        print(e)
        print(pt_out)
        print(gm_out)
