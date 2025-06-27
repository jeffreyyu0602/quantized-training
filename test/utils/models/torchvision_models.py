import torch

from torchvision import models, transforms
from tqdm import tqdm

from quantized_training import (
    DerivedQuantizationSpec,
    FusedAmaxObsFakeQuantize,
    QuantizationConfig,
    QuantizationSpec,
    convert_pt2e,
    export_model,
    prepare_pt2e,
    transform,
    compile,
    derive_bias_qparams_fn,
    extract_input_preprocessor,
    fuse,
)
from quantized_training.codegen.utils import get_conv_bn_layers

from .utils import get_transform_args, get_compile_args

def load_model(args):
    if args.model_name_or_path is None:
        args.model_name_or_path = "DEFAULT"

    try:
        model = models.__dict__[args.model](weights=args.model_name_or_path).eval()
    except Exception as e:
        model = models.__dict__[args.model](pretrained=True).eval()

        if args.model_name_or_path:
            checkpoint = torch.load(args.model_name_or_path, map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'], strict=False)

    if args.bf16:
        model.bfloat16()
    return model

def quantize_and_dump_model(model, quantizer, calibration_data, vector_stages, args):
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    transform_args = get_transform_args(args, vector_stages)
    compile_args = get_compile_args(args)

    modules_to_fuse = get_conv_bn_layers(model)
    if len(modules_to_fuse) > 0:
        model = torch.ao.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

    # Accelerator only supports 2x2 maxpool
    if args.use_maxpool_2x2:
        for module in model.modules():
            if isinstance(module, torch.nn.MaxPool2d):
                module.kernel_size = 2
                module.stride = 2
                module.padding = 0

    if "mobilenet" in args.model:
        quantizer.set_module_name("classifier", None)
    else:
        quantizer.set_module_name("fc", None)

    # use per-tensor instead of microscaling for conv1 in resnet18 and resnet50
    if args.activation is not None and "microscaling" in args.activation:
        qspec = QuantizationSpec.from_str("int8,qs=per_tensor_symmetric")
        qspec.observer_or_fake_quant_ctr = FusedAmaxObsFakeQuantize

        bias_qspec = DerivedQuantizationSpec(
            derived_from=None,
            derive_qparams_fn=derive_bias_qparams_fn,
            dtype=None,
        )

        qconfig = QuantizationConfig(qspec, None, qspec, bias_qspec)
        quantizer.set_module_name("^conv1$", qconfig)

    example_args = (torch.randn(1, 3, 224, 224, dtype=torch_dtype),)
    gm = export_model(model, example_args)

    from collections import OrderedDict
    def run_and_record_nodes(gm, *example_args):
        from collections import OrderedDict
        env = {}
        node_outputs = OrderedDict()

        args_iter = iter(example_args)

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                env[node.name] = next(args_iter)

            elif node.op == "get_attr":
                # Resolve parameter or buffer from gm
                attr_itr = node.target.split('.')
                attr_val = gm
                for attr in attr_itr:
                    attr_val = getattr(attr_val, attr)
                env[node.name] = attr_val

            elif node.op == "call_function":
                args = [env[arg.name] if hasattr(arg, "name") else arg for arg in node.args]
                kwargs = {k: env[v.name] if hasattr(v, "name") else v for k, v in node.kwargs.items()}
                out = node.target(*args, **kwargs)
                env[node.name] = out
                node_outputs[node.name] = out

            elif node.op == "call_method":
                self_obj = env[node.args[0].name]
                method_args = [env[a.name] if hasattr(a, "name") else a for a in node.args[1:]]
                out = getattr(self_obj, node.target)(*method_args)
                env[node.name] = out
                node_outputs[node.name] = out

            elif node.op == "call_module":
                submod = dict(gm.named_modules())[node.target]
                mod_args = [env[a.name] if hasattr(a, "name") else a for a in node.args]
                mod_kwargs = {k: env[v.name] if hasattr(v, "name") else v for k, v in node.kwargs.items()}
                out = submod(*mod_args, **mod_kwargs)
                env[node.name] = out
                node_outputs[node.name] = out

            elif node.op == "output":
                def unwrap(v):
                    if hasattr(v, "name"):
                        return env[v.name]
                    elif isinstance(v, (list, tuple)):
                        return type(v)(unwrap(x) for x in v)
                    else:
                        return v

                output_val = unwrap(node.args[0])
                node_outputs["output"] = output_val

        return node_outputs
    node_out1 = run_and_record_nodes(gm, *example_args)

    def match_and_rewrite(source_fn, args, kwargs):
        print(f"match_and_rewrite called with source_fn: {source_fn}")
        if source_fn not in [torch.nn.Conv2d, torch.nn.functional.conv2d]:
            print("Not matching Conv2d, returning None")
            return None
        
        weight = args[1]
        if weight.shape[1] != 3 or weight.shape[2] != 3 or weight.shape[3] != 3:
            print("Not matching non 3x3x3 weight, returning None")
            return None

        print(weight.shape)

        import torch.nn as nn
        import torch.nn.functional as F

        class Padded_3C(nn.Module):
            def __init__(self):
                super().__init__()
                self.stride = 2

            # NOTE: replacement module has to be a stateless module, meanining that
            # it cannot have any parameters or buffers. All parameters and buffers
            # should be passed as arguments to the forward method.
            def forward(self, x, weight, bias=None):
                print("Padded_3C.forward called")
                B, C, H, W = x.shape
                # x_padded = F.pad(x, (0, 4, 0, 4))
                weight_padded = F.pad(weight, (2, 2, 2, 2))
                result = F.conv2d(x, weight_padded, bias=bias, stride=self.stride, groups=1, padding=3)
                return result
            
        return Padded_3C

    # NOTE: rewrite_fx_graph needs to be called before prepare_pt2e. This
    # is temporary, and we will fix it in the future.
    from quantized_training import rewrite_fx_graph
    rewrite_fx_graph(gm, match_and_rewrite)

    node_out2 = run_and_record_nodes(gm, *example_args)
    def generate_diff_report(node_out1, node_out2, atol=1e-4):
        report = []
        debug_layer = False
        for k in node_out1:
            if k == 'conv2d':
                k2 = 'conv2d_52'
            else:
                k2 = k
            if k2 not in node_out2:
                report.append(f"Node '{k2}' missing in second model.")
                continue

            v1, v2 = node_out1[k], node_out2[k2]

            if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
                if not torch.allclose(v1, v2, atol=atol):
                    diff = (v1 - v2).abs().max().item()
                    if (k2 == 'hardtanh_' or k2 == 'conv2d_1' or k2 == 'hardtanh__1' or k2 == 'conv2d_2' or k2 == 'conv2d_3' or k2 == 'hardtanh__2')\
                        and debug_layer:
                        for x1 in range(v1.shape[0]):
                            for x2 in range(v1.shape[1]):
                                for x3 in range(v1.shape[2]):
                                    for x4 in range(v1.shape[3]):
                                        if v1[x1, x2, x3, x4] != v2[x1, x2, x3, x4]:
                                            report.append(f"Mismatch at node '{k}': max abs diff = {diff:.6f}, "
                                                        f"shape = {v1.shape}, index = ({x1}, {x2}, {x3}, {x4})")
                                        # else:
                                        #     report.append(f"Match at node '{k}': value = {v1[x1, x2, x3, x4]}, index = ({x1}, {x2}, {x3}, {x4})")

                    report.append(f"Mismatch at node '{k}': max abs diff = {diff:.6f}, shape = {v1.shape}")
            elif isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
                if len(v1) != len(v2):
                    report.append(f"Mismatch at node '{k}': list/tuple length differs ({len(v1)} vs {len(v2)})")
                    continue
                for i, (item1, item2) in enumerate(zip(v1, v2)):
                    if not torch.allclose(item1, item2, atol=atol):
                        diff = (item1 - item2).abs().max().item()
                        report.append(f"Mismatch at node '{k}[{i}]': max abs diff = {diff:.6f}, shape = {item1.shape}")
            else:
                report.append(f"Node '{k}': unsupported types ({type(v1)} vs {type(v2)})")

        if not report:
            report.append("All nodes match!")

        return "\n".join(report)

    diff_report = generate_diff_report(node_out1, node_out2)
    print(diff_report)

    gm = prepare_pt2e(gm, quantizer, example_args)

    for i in tqdm(range(10), desc=f"Calibrating {model.__class__.__name__}"):
        inputs = calibration_data[i]["image"]
        with torch.no_grad():
            gm(inputs.to(torch_dtype))

    convert_pt2e(gm, args.bias)

    old_output = gm(*example_args)

    transform(gm, example_args, **transform_args, fuse_operator=False)

    gm, preprocess_fn = extract_input_preprocessor(gm)
    example_args = (preprocess_fn(*example_args),)

    fuse(gm, vector_stages, example_args)

    gm.graph.print_tabular()

    new_output = gm(*example_args)

    compile(gm, example_args, **compile_args)
    return gm, old_output, new_output, preprocess_fn

def evaluate(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for image_label_pair in tqdm(dataset, desc=f"Evaluating {model.__class__.__name__}"):
            # for running the original model without the preprocessing function 
            # applied to the dataset
            image = image_label_pair["image"].to(device)
            label = image_label_pair["label"]
    
            logits = model(image)
            prediction = torch.argmax(logits, dim=-1)
            if prediction.item() == label:
                correct_predictions += 1
            total_samples += 1

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"{model.__class__.__name__} Accuracy: {accuracy:.4f}")
