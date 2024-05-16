import argparse
import logging

import torch
from torch.export import export
from transformers import AutoModelForSemanticSegmentation, AutoModelForSequenceClassification

from quantized_training.codegen import ShapeProp

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


def transform(model, example_args, generate_graph=False, output_file="compute_graph"):
    exported_program: torch.export.ExportedProgram = \
        export(model, example_args)

    # print(exported_program)
    exported_program.graph.print_tabular()

    shape_prop = ShapeProp(exported_program.module())
    shape_prop.transform()
    # shape_prop.graph.print_tabular()

    pt_out = model(*example_args)
    gm_out = shape_prop.propagate(*example_args)[0][0]

    params = shape_prop.gen_code()
    with open('params.pb', 'wb') as f:
        f.write(params.SerializeToString())

    import json
    from google.protobuf.json_format import MessageToDict

    data = MessageToDict(params)
    print(json.dumps(data, indent=4))

    if generate_graph:
        shape_prop.gen_compute_graph(output_file)

    return (shape_prop, pt_out, gm_out)


if __name__ == "__main__":
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="resnet50")
    parser.add_argument("--generate_graph", action="store_true")
    args = parser.parse_args()

    if "resnet50" in args.models:
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.eval()
        
        module_names = [name for name, _ in model.named_modules()]
        modules_to_fuse = _pair_conv_bn(module_names)
        model = torch.ao.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

        example_args = (torch.randn(1, 3, 224, 224),)
        transform(model, example_args, args.generate_graph, "resnet50")

    if "segformer" in args.models:
        replace_interpolate()
        model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        example_args = (torch.randn(1, 3, 512, 672),)
        _, pt_out, gm_out = transform(model, example_args, args.generate_graph, "segformer")
        torch.set_printoptions(precision=8)
        diff = pt_out.logits != gm_out
        print(torch.sum(diff))
        print(pt_out.logits[diff])
        print(gm_out[diff])
        assert torch.all(pt_out.logits == gm_out), "Outputs do not match"

    if "mobilebert" in args.models:
        model = AutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased")
        example_args = (torch.randint(0, 30522, (1, 128), dtype=torch.long),)
        _, pt_out, gm_out = transform(model, example_args, args.generate_graph, "mobilebert")
        assert torch.all(pt_out.logits == gm_out), "Outputs do not match"