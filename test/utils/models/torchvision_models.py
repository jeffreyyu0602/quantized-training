import re
import torch

from torchvision import models
from tqdm import tqdm

from quantized_training import (
    DerivedQuantizationSpec,
    FusedAmaxObsFakeQuantize,
    QuantizationConfig,
    QuantizationSpec,
    convert_pt2e,
    prepare_pt2e,
    transform,
    compile,
    derive_bias_qparams_fn,
    extract_input_preprocessor,
    fuse,
)
from quantized_training.codegen import get_conv_bn_layers

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

        if args.activation is not None and "microscaling" in args.activation:
            mobilenet_int8_layers = [
                "features.0.0",
                # "features.1.conv.0.0",
                # "features.2.conv.1.0",
                # "features.3.conv.1.0",
                # "features.4.conv.1.0",
                # "features.5.conv.1.0",
                # "features.6.conv.1.0",
                # "features.7.conv.1.0",
                # "features.8.conv.1.0",
                # "features.9.conv.1.0",
                # "features.10.conv.1.0",
                # "features.11.conv.1.0",
                # "features.12.conv.1.0",
                # "features.13.conv.1.0",
                # "features.14.conv.1.0",
                # "features.15.conv.1.0",
                # "features.16.conv.1.0",
                # "features.17.conv.1.0",
            ]

            qspec = QuantizationSpec.from_str("int8,qs=per_tensor_symmetric")
            qspec.observer_or_fake_quant_ctr = FusedAmaxObsFakeQuantize

            bias_qspec = DerivedQuantizationSpec(
                derived_from=None,
                derive_qparams_fn=derive_bias_qparams_fn,
                dtype=None,
            )

            qconfig = QuantizationConfig(qspec, None, qspec, bias_qspec)

            for layer in mobilenet_int8_layers:
                quantizer.set_module_name(layer, qconfig)

        model.features[0][0].padding = (3, 3)
        model.features[0][0].weight.data = torch.nn.functional.pad(model.features[0][0].weight.data, (2, 2, 2, 2))

    # Some designs do not support quantized fc layers
    if not args.quantize_fc:
        quantizer.set_module_name("fc", None)

    if args.residual is not None:
        qspec = QuantizationSpec.from_str(f"{args.residual},qs=per_tensor_symmetric")
        qconfig = QuantizationConfig(qspec, None, None, None)
        quantizer.set_object_type(torch.ops.aten.add.Tensor, qconfig)
        quantizer.set_object_type(torch.ops.aten.add_.Tensor, qconfig)

    # use per-tensor instead of microscaling for conv1 in resnet18 and resnet50
    if args.activation is not None and "microscaling" in args.activation:
        dtype = args.activation.split(",")[0]
        if (match := re.fullmatch(r'nf(\d+)(?:_(\d+))?', dtype)):
            bits = int(match.group(1))
            dtype = f"int{bits}"
        qspec = QuantizationSpec.from_str(f"{dtype},qs=per_tensor_symmetric")
        qspec.observer_or_fake_quant_ctr = FusedAmaxObsFakeQuantize

        bias_qspec = DerivedQuantizationSpec(
            derived_from=None,
            derive_qparams_fn=derive_bias_qparams_fn,
            dtype=None,
        )

        qconfig = QuantizationConfig(qspec, None, qspec, bias_qspec)
        quantizer.set_module_name("^conv1$", qconfig)

    example_args = (torch.randn(1, 3, 224, 224, dtype=torch_dtype),)
    gm = prepare_pt2e(model, quantizer, example_args)

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
