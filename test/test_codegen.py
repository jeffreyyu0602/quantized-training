import argparse
import copy
import operator
import os

import torch
from datasets import load_dataset
from google.protobuf import text_format
from google.protobuf.json_format import MessageToJson
from torch._export import capture_pre_autograd_graph
from torch.utils.data import DataLoader
from torchvision import models
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSemanticSegmentation,
    AutoImageProcessor,
    AutoTokenizer,
    default_data_collator,
)
from tqdm import tqdm

from quantized_training import (
    add_qspec_args,
    convert_pt2e,
    get_default_quantizer,
    prepare_pt2e,
)
from quantized_training.codegen import (
    MemoryManager,
    ShapeProp,
    allocate_activations,
    allocate_weights,
    fuse_operator,
    gen_code,
    gen_compute_graph,
    split_multi_head_attention,
)
from quantized_training.quantize_pt2e import _fuse_quantize_dequantize_with_previous_op
from quantized_training.quantizer.xnnpack_quantizer_utils import _convert_scalars_to_attrs


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

OPERATOR_MAPPINGS = {
    "add": ["add", "add_", operator.add, torch.add, operator.iadd],
    "sub": ["sub", "sub_", operator.sub, torch.sub, operator.isub],
    "mul": ["mul", "mul_", operator.mul, torch.mul, operator.imul],
    "div": ["div", "div_", operator.truediv, torch.div, operator.itruediv],
    "exp": [torch.exp],
    "relu": [torch.nn.ReLU, torch.nn.functional.relu, torch.nn.functional.relu_],
    "gelu": [torch.nn.GELU, torch.nn.functional.gelu],
    "gemm": [
        torch.nn.Conv2d,
        torch.nn.Linear,
        torch.matmul,
        torch.ops.quantized_ops.conv2d_mx.default,
        torch.ops.quantized_ops.linear_mx.default,
        torch.ops.quantized_ops.matmul_mx.default,
    ],
    "quantize": [torch.ops.quantized_ops.quantize_symmetric],
    "dequantize": [torch.ops.quantized_ops.dequantize_symmetric],
}


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


def flatten_args(mixed_list):
    flattened_list = []
    for element in mixed_list:
        if isinstance(element, list):
            flattened_list.extend(flatten_args(element))
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

    uplifted_args = flatten_args(list(example_args)) + list(example_kwargs.values())

    ShapeProp(gm).propagate(*uplifted_args)
    split_multi_head_attention(gm)
    ShapeProp(gm).propagate(*uplifted_args)

    _fuse_quantize_dequantize_with_previous_op(gm)

    pipeline = {
        0: ["gemm"],
        1: ["dequantize"],
        2: ["add", "sub", "mul", "div"],
        3: ["exp"],
        4: ["add", "mul", "div"],
        5: ["relu"],
        6: ["quantize"],
    }

    # If there is no corresponding mapping, we directly append the op string
    pipeline = {
        stage: [item for op in ops for item in OPERATOR_MAPPINGS.get(op, [op])]
        for stage, ops in pipeline.items()
    }

    fuse_operator(gm, pipeline)
    gm.graph.print_tabular()

    pt_out = model(*example_args, **example_kwargs)
    gm_out = gm(*example_args, *list(example_kwargs.values()))

    ShapeProp(gm).propagate(*uplifted_args)

    manager = MemoryManager(1024 ** 4)
    allocate_weights(gm, manager)
    allocate_activations(gm, manager)

    manager.print_partitions()
    print("\nMemory allocated to tensors:")
    for node in gm.graph.nodes:
        if (partition := node.meta.get('memory', None)) is None:
            print(f"Node {node.name} does not have memory allocated")
            continue
        print(f"{node.name}: {partition.start}, {partition.end}")

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


TORCHVISION_MODELS = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "mobilenet": models.mobilenet_v2,
}


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=10)

    parser = argparse.ArgumentParser()
    parser.add_argument("model", default="resnet50")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--task_name",
        default="sst2",
        help="Name of the task to load the dataset"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for generated tensor files"
    )
    add_qspec_args(parser)
    args = parser.parse_args()

    quantizer = get_default_quantizer(
        args.activation, args.output_activation, args.weight, args.bias
    )

    if args.model in TORCHVISION_MODELS:
        model = TORCHVISION_MODELS[args.model](pretrained=True).eval()

        if args.model_name_or_path:
            checkpoint = torch.load(args.model_name_or_path, map_location="cpu")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if args.bf16:
            model.bfloat16()
        torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

        module_names = [name for name, _ in model.named_modules()]
        modules_to_fuse = _pair_conv_bn(module_names)
        if len(modules_to_fuse) > 0:
            model = torch.ao.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

        # Accelerator only supports 2x2 maxpool
        for module in model.modules():
            if isinstance(module, torch.nn.MaxPool2d):
                module.kernel_size = 2
                module.stride = 2
                module.padding = 0

        quantizer.set_module_name("fc", None)

        example_args = (torch.randn(1, 3, 224, 224, dtype=torch_dtype),)
        model = prepare_pt2e(model, quantizer, example_args)

        dataset = load_dataset("zh-plus/tiny-imagenet")

        image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

        for i in tqdm(range(10)):
            inputs = image_processor(dataset['train'][i]["image"], return_tensors="pt")
            with torch.no_grad():
                model(inputs.pixel_values.to(torch_dtype))

        convert_pt2e(model)

        pt_out, gm_out = transform(
            model,
            example_args,
            output_file=args.model,
            output_dir=args.output_dir,
        )
    elif args.model == "segformer":
        replace_interpolate()

        if args.model_name_or_path is None:
            args.model_name_or_path = "nvidia/segformer-b0-finetuned-ade-512-512"

        model = AutoModelForSemanticSegmentation.from_pretrained(args.model_name_or_path).eval()

        modules_to_fuse = ["decode_head.linear_fuse", "decode_head.batch_norm"]
        model = torch.ao.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

        example_args = (torch.randn(1, 3, 512, 672),)

        model = prepare_pt2e(model, quantizer, example_args)
        convert_pt2e(model)

        pt_out, gm_out = transform(
            model,
            example_args,
            output_file="segformer",
            output_dir=args.output_dir,
        )
        pt_out = pt_out.logits
        gm_out = gm_out.logits
    elif args.model == "mobilebert":
        if args.model_name_or_path is None:
            args.model_name_or_path = "google/mobilebert-uncased"
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path).eval()

        input_ids = torch.randint(0, 30522, (1, 128), dtype=torch.long)
        input_shape = input_ids.size()
        attention_mask = torch.ones(input_shape)
        token_type_ids = torch.zeros(input_shape, dtype=torch.long)
        position_ids = torch.ones((1, 128), dtype=torch.long)
        head_mask = None

        if args.bf16:
            model.bfloat16()

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

        class MobileBertNoEmbed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mobilebert = model.mobilebert
                self.classifier = model.classifier

            def forward(self, *args, **kwargs):
                hidden_states = self.mobilebert.encoder(*args, **kwargs)[0]
                first_token_tensor = hidden_states[:, 0]
                output = self.classifier(first_token_tensor)
                return output

        gm = prepare_pt2e(MobileBertNoEmbed(), quantizer, example_args)

        # calibration
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        raw_datasets = load_dataset("glue", args.task_name)

        sentence1_key, sentence2_key = task_to_keys[args.task_name]

        def preprocess_function(examples):
            # Tokenize the texts
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*texts, padding="max_length", max_length=128, truncation=True)
            result["labels"] = examples["label"]
            return result

        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

        train_dataset = processed_datasets["train"]

        train_dataloader = DataLoader(train_dataset, collate_fn=default_data_collator, batch_size=1)

        for step, batch in enumerate(tqdm(train_dataloader)):
            embedding_output = model.mobilebert.embeddings(
                input_ids=batch["input_ids"],
                token_type_ids=batch["token_type_ids"]
            )
            extended_attention_mask = model.mobilebert.get_extended_attention_mask(batch["attention_mask"], input_shape)
            gm(embedding_output, extended_attention_mask, head_mask)

            if step == 9:
                break

        convert_pt2e(gm)

        pt_out, gm_out = transform(
            gm,
            example_args,
            output_file="mobilebert",
            output_dir=args.output_dir,
        )
    elif args.model == "mobilebert_encoder":
        if args.model_name_or_path is None:
            args.model_name_or_path = "google/mobilebert-uncased"
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path).eval()

        if args.bf16:
            model.bfloat16()
        torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

        example_args = (
            torch.randn(1, 128, 512, dtype=torch_dtype),
            torch.ones(1, 128, 128, dtype=torch_dtype),
            None,
        )

        class MobileBertEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, *args, **kwargs):
                output = model.mobilebert.encoder.layer[0](*args, **kwargs)
                return output[0][0]

        gm = prepare_pt2e(MobileBertEncoder(), quantizer, example_args)

        # Generate a random scale, otherwise a scale of 1 will be optimized away.
        for name, module in gm.named_modules():
            if hasattr(module, "scale"):
                module.scale = torch.randn_like(module.scale)

        convert_pt2e(gm)

        pt_out, gm_out = transform(
            gm,
            example_args,
            output_file="mobilebert",
            output_dir=args.output_dir,
        )

        pt_out = pt_out[0]
        gm_out = gm_out[0]
    elif args.model == "bert":
        if args.model_name_or_path is None:
            args.model_name_or_path = "bert-base-uncased"
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path).eval()

        input_ids = torch.randint(0, 30522, (1, 128), dtype=torch.long)
        input_shape = input_ids.size()
        attention_mask = torch.ones(input_shape)
        token_type_ids = torch.zeros(input_shape, dtype=torch.long)
        position_ids = torch.ones((1, 128), dtype=torch.long)
        head_mask = None

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = model.bert.get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = model.bert.get_head_mask(head_mask, model.config.num_hidden_layers)

        class BertNoEmbed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = model.bert
                self.classifier = model.classifier

            def forward(self, *args, **kwargs):
                hidden_states = self.bert.encoder(*args, **kwargs)[0]
                first_token_tensor = hidden_states[:, 0]
                output = self.classifier(first_token_tensor)
                return output

        embedding_output = model.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        example_args = (embedding_output, extended_attention_mask, head_mask)

        gm = prepare_pt2e(BertNoEmbed(), quantizer, example_args)
        convert_pt2e(gm)

        pt_out, gm_out = transform(
            gm,
            example_args,
            output_file="bert",
            output_dir=args.output_dir,
        )
    else:
        raise ValueError(f"Model {args.model} not supported")

    try:
        assert torch.all(pt_out == gm_out)
        print("Results match")
    except Exception as e:
        print(e)
        print(pt_out)
        print(gm_out)
