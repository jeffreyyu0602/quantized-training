import argparse
import copy
import operator
import os
import logging
from typing import List

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
    DerivedQuantizationSpec,
    FusedAmaxObsFakeQuantize,
    MemoryManager,
    QuantizationConfig,
    QuantizationSpec,
    ShapeProp,
    add_qspec_args,
    allocate_activations,
    allocate_weights,
    convert_pt2e,
    get_aten_graph_module,
    fuse_operator,
    gen_code,
    gen_compute_graph,
    get_default_quantizer,
    prepare_pt2e,
    split_multi_head_attention,
    visualize_memory_layout,
)
from quantized_training.quantize_pt2e import _fuse_quantize_dequantize_with_previous_op
from quantized_training.quantizer.xnnpack_quantizer_utils import _convert_scalars_to_attrs

logger = logging.getLogger(__name__)


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

quantized_ops = torch.ops.quantized_ops
OPERATOR_MAPPINGS = {
    "gemm": [torch.nn.Conv2d, torch.nn.Linear, torch.matmul, operator.matmul],
    "add": ["add", "add_", operator.add, torch.add, operator.iadd],
    "sub": ["sub", "sub_", operator.sub, torch.sub, operator.isub],
    "mul": ["mul", "mul_", operator.mul, torch.mul, operator.imul],
    "div": ["div", "div_", operator.truediv, torch.div, operator.itruediv],
    "relu": [torch.nn.ReLU, torch.nn.functional.relu, torch.nn.functional.relu_],
    "gelu": [torch.nn.GELU, torch.nn.functional.gelu],
    "maxpool2d": [torch.nn.MaxPool2d, torch.nn.functional.max_pool2d],
    "avgpool2d": [torch.nn.AdaptiveAvgPool2d, torch.nn.functional.adaptive_avg_pool2d],
    "quantize": [quantized_ops.quantize],
    "dequantize": [quantized_ops.dequantize],
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


def get_conv_bn_layers(model):
    layers = []
    module_names = list(model._modules)
    for k, name in enumerate(module_names):
        if len(list(model._modules[name]._modules)) > 0:
            conv_bn_pairs = get_conv_bn_layers(model._modules[name])
            layers.extend([[f'{name}.{conv}', f'{name}.{bn}'] for conv, bn in conv_bn_pairs])
        else:
            if isinstance(model._modules[name], torch.nn.BatchNorm2d):
                if isinstance(model._modules[module_names[k-1]], torch.nn.Conv2d):
                    layers.append([module_names[k-1], name])
    return layers


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

    from torch.utils._pytree import tree_flatten
    flatten_args, spec = tree_flatten((example_args, example_kwargs))

    ShapeProp(gm).propagate(*flatten_args)
    split_multi_head_attention(gm)

    ShapeProp(gm).propagate(*flatten_args)

    _fuse_quantize_dequantize_with_previous_op(gm)

    pipeline = {
        0: ["gemm"],
        1: ["dequantize"],
        2: ["add", "sub", "mul", "div"],
        3: ["exp"],
        4: ["add", "mul", "div"],
        5: ["relu"],
        7: ["quantize"],
    }

    # If there is no corresponding mapping, we directly append the op string
    pipeline = {
        stage: [item for op in ops for item in OPERATOR_MAPPINGS.get(op, [op])]
        for stage, ops in pipeline.items()
    }

    from quantized_training.codegen.mapping import optimize_graph

    custom_mapping = [torch.ops.aten.softmax.int]
    optimize_graph(gm, pipeline, custom_mapping)

    fuse_operator(gm, pipeline)
    gm.graph.print_tabular()

    pt_out = model(*example_args, **example_kwargs)
    gm_out = gm(*example_args, *list(example_kwargs.values()))

    ShapeProp(gm).propagate(*flatten_args)

    try:
        manager = MemoryManager(1024 ** 4)
        allocate_weights(gm, manager)
        manager = MemoryManager(1024 ** 4)
        allocate_activations(gm, manager)
    except RuntimeError:
        manager.print_partitions()
        filename = os.path.join(output_dir, "memory_layout.png")
        visualize_memory_layout(manager.snapshots, filename)
        logger.error(f"Memory allocation failed. Memory layout saved to {filename}")

    manager.print_partitions()
    print("\nMemory allocated to tensors:")
    for node in gm.graph.nodes:
        if (partition := node.meta.get('memory', None)) is None:
            print(f"Node {node.name} does not have memory allocated")
            continue
        print(f"{node.name}: {partition.start}, {partition.end}")

    params = gen_code(
        gm,
        flatten_args,
        os.path.join(output_dir, "tensor_files")
    )

    with open(os.path.join(output_dir, 'params.pb'), 'wb') as f:
        f.write(params.SerializeToString())

    with open(os.path.join(output_dir, 'params.txt'), "w") as f:
        f.write(text_format.MessageToString(params))

    with open(os.path.join(output_dir, 'params.json'), "w") as f:
        f.write(MessageToJson(params))

    layers = [p.name for p in params.params if p.WhichOneof("param_type") != "nop"]
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
    parser.add_argument(
        "--use_mixed_qscheme",
        action="store_true",
        help="Quantize attention matrix multiplication using per-tensor symmetric quantization"
    )
    add_qspec_args(parser)
    args = parser.parse_args()

    quantizer = get_default_quantizer(
        input_activation=args.activation,
        output_activation=args.output_activation,
        weight=args.weight,
        bias=args.bias,
        force_scale_power_of_two=args.force_scale_power_of_two,
    )

    if args.use_mixed_qscheme:
        qspec = QuantizationSpec.from_str("int8,qs=per_tensor_symmetric")
        qspec.observer_or_fake_quant_ctr = FusedAmaxObsFakeQuantize

        qconfig = QuantizationConfig(qspec, None, qspec, None)
        quantizer.set_object_type(torch.ops.aten.matmul.default, qconfig)

        from quantized_training.quantize_pt2e import derive_bias_qparams_fn

        bias_qspec = DerivedQuantizationSpec(
            derived_from=None,
            derive_qparams_fn=derive_bias_qparams_fn,
            dtype=None,
        )

        qconfig = QuantizationConfig(qspec, None, qspec, bias_qspec)
        quantizer.set_module_name("conv1", qconfig)

    if args.model in TORCHVISION_MODELS:
        model = TORCHVISION_MODELS[args.model](pretrained=True).eval()

        if args.model_name_or_path:
            checkpoint = torch.load(args.model_name_or_path, map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'], strict=False)

        if args.bf16:
            model.bfloat16()
        torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

        modules_to_fuse = get_conv_bn_layers(model)
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

        quantizer.set_module_name("classifier", None)

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

        convert_pt2e(gm, args.bias)

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
    elif args.model == "llama":
        from transformers import AutoModelForCausalLM

        if args.model_name_or_path is None:
            args.model_name_or_path = "meta-llama/Llama-3.2-1B"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager", # turn off flash attention
        )

        input_ids = torch.randint(0, 30522, (1, 128), dtype=torch.long)
        target_ids = input_ids.clone()

        inputs_embeds = model.model.embed_tokens(input_ids)
        cache_position = torch.arange(0, inputs_embeds.shape[1])
        position_ids = cache_position.unsqueeze(0)
        causal_mask = model.model._update_causal_mask(
            None, inputs_embeds, cache_position, None, None
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

        example_args = (hidden_states, causal_mask, position_embeddings)

        class LlamaWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model.model
                self.lm_head = model.lm_head

            def forward(self, hidden_states, causal_mask, position_embeddings):
                for decoder_layer in self.model.layers:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_embeddings=position_embeddings,
                    )
                    hidden_states = layer_outputs[0]
                logits = self.lm_head(hidden_states)
                return logits

        gm = prepare_pt2e(LlamaWrapper(), quantizer, example_args)
        convert_pt2e(gm)

        # Replace LLaMA RMSNorm with ATen layer_norm
        original_graph = gm.graph

        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        layer_norm = LlamaRMSNorm(2048).bfloat16()

        layer_norm_args = (torch.randn(1, 128, 2048, dtype=torch.bfloat16),)
        pattern = get_aten_graph_module(layer_norm, layer_norm_args)
        pattern = quantizer.transform_for_annotation(pattern)
        pattern_graph = pattern.graph

        from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher

        matcher = SubgraphMatcher(
            pattern_graph,
            match_output=False,
            match_placeholder=False,
            remove_overlapping_matches=True,
            ignore_literals=False,
        )
        _matches: List[InternalMatch] = matcher.match(original_graph)
        print(f"Found {len(_matches)} matches")

        param_node = next(iter(n for n in pattern_graph.nodes if n.name == "_param_constant0"))

        for match in _matches:
            mapped_param_node = match.nodes_map[param_node]
            layer_norm_inputs = [match.placeholder_nodes[0], [2048], mapped_param_node]
            output_node = match.returning_nodes[0]
            with original_graph.inserting_before(output_node):
                new_node = original_graph.call_function(
                    torch.ops.aten.layer_norm.default,
                    tuple(layer_norm_inputs),
                    {}
                )
            output_node.replace_all_uses_with(new_node)
            original_graph.erase_node(output_node)
 
        original_graph.lint()
        original_graph.eliminate_dead_code()
        gm.recompile()

        pt_out, gm_out = transform(
            gm,
            example_args,
            output_dir=args.output_dir,
        )
    elif args.model == "llama_encoder":
        pass
    elif args.model == "test":
        class TestModel(torch.nn.Module):
            def forward(self, x):
                # outputs = torch.split_with_sizes(x, (5, 5))
                # outputs = torch.ops.aten.split.sizes(x, (5, 5))
                outputs = torch.split(x, (5, 5))
                print(outputs)
                outputs = torch.median(outputs[0])
                return outputs

        example_args = (torch.randn(10),)

        TestModel()(*example_args)

        gm = prepare_pt2e(TestModel(), quantizer, example_args)
        convert_pt2e(gm)

        pt_out, gm_out = transform(
            gm,
            example_args,
            output_dir=args.output_dir,
        )
    elif args.model == "mcunet":
        import sys
        sys.path.append("mcunet")
        from mcunet.model_zoo import net_id_list, build_model
        print(net_id_list)  # the list of models in the model zoo

        # pytorch fp32 model
        model, image_size, description = build_model(net_id="mcunet-in4", pretrained=True)  # you can replace net_id with any other option from net_id_list

        sys.path.append(".")
        from bn_folder import bn_folding_model
        model = bn_folding_model(model.eval())
        model.bfloat16()
        example_args = (torch.randn(1, 3, 160, 160, dtype=torch.bfloat16),)

        gm = prepare_pt2e(model, quantizer, example_args)
        convert_pt2e(gm)

        pt_out, gm_out = transform(
            gm,
            example_args,
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
