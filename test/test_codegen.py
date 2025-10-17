import argparse
import logging
import os
import re
import sys

import torch
from datasets import load_dataset
from torchvision import models, transforms
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    StaticCache,
)
from tqdm import tqdm

from quantized_training import (
    TorchExportableModuleWithStaticCache,
    QuantizationConfig,
    QuantizationSpec,
    add_qspec_args,
    compile,
    convert_and_export_with_split_cache,
    convert_pt2e,
    extract_input_preprocessor,
    fuse,
    get_default_quantizer,
    prepare_pt2e,
    sink_obs_or_fq,
    swap_llama_attention,
    transform,
)
from quantized_training.codegen.utils import (
    remove_autocast_nodes,
    replace_rmsnorm_with_layer_norm,
    strip_softmax_dtype,
)
from quantized_training.llm_utils import fuse_dequantize_quantize

from utils.models import bert, mobilebert, torchvision_models, vit
from utils.dataset import glue, imagenet


script_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(script_dir, '../examples/language_modeling')
sys.path.append(os.path.abspath(target_path))

from prepare_model import set_qconfig

logger = logging.getLogger()


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


vector_stages = [
    [
        ["gemm"],
        ["dequantize"],
        ["add", "sub", "mul", "div"],
        ["exp", "abs", "relu", "gelu", "tanh", "silu", "vmap", "hardtanh"],
        ["add", "mul", "div"],
        ["div", "quantize"],
    ],
    [
        ["layer_norm", "softmax"],
        ["quantize"],
    ]
]


def get_llm_qscheme(bs=64, threshold=None):
    outlier = f",outlier={threshold}" if threshold is not None else ""
    return {
        r"self_attn\.q_proj$": [
            f"int6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3" + outlier,
            f"int2,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
        ],
        r"self_attn\.k_proj$": [
            f"nf4_6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3" + outlier,
            f"int2,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
        ],
        r"self_attn\.v_proj$": [
            f"int6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3" + outlier,
            f"nf4_6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
        ],
        r"self_attn\.o_proj$": [
            f"nf4_6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
            f"int6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
        ],
        torch.nn.Linear: [
            f"nf4_6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3" + outlier,
            f"nf4_6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
        ],
        torch.ops.aten.matmul.default: [
            f"int6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
            f"int6,qs=microscaling,bs={bs},ax=-2,scale=fp8_e5m3",
        ],
        (r"lm_head", torch.ops.aten.linear.default, 0): [
            f"int6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
            f"nf4_6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
        ],
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False, precision=10)
    torch.set_num_threads(32)

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
        "--model_output_dir",
        required=True,
        help="Output directory for generated tensor files"
    )
    parser.add_argument(
        "--dump_dataset",
        action="store_true",
        help="Whether to dump the dataset to disk for later use."
    )
    parser.add_argument(
        "--dump_verification_file",
        action="store_true",
        help="Whether to save tensors for verification."
    )
    parser.add_argument(
        "--dataset_output_dir",
        help="Output directory for dataset files"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=512,
        help="Context length for the LLM decoding."
    )
    parser.add_argument(
        "--remove_duplicate",
        action="store_true",
        help="Only compiler for a single encoder/decoder layer in Transformer models."
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Quantization scheme to use for LLMs."
    )
    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=None,
        help="Whether to filter outliers when quantizing activations."
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=None,
        help="Total L2 SRAM size in SoC."
    )
    parser.add_argument(
        "--num_banks",
        type=int,
        default=None,
        help="Number of banks in the accelerator."
    )
    parser.add_argument(
        "--transpose_weight",
        action="store_true",
        help="Whether to transpose weights for executing on the accelerator."
    )
    parser.add_argument(
        "--transpose_fc",
        action="store_true",
        help="Whether to transpose the weight of the fully connected layer."
    )
    parser.add_argument(
        "--use_maxpool_2x2",
        action="store_true",
        help="Whether to use 2x2 maxpool for resnet18 and resnet50."
    )
    parser.add_argument(
        "--conv2d_im2col",
        action="store_true",
        help=(
            "Whether to use transform conv2d operation with input channel "
            "dimension smaller than unroll to linear operation through im2col."
        )
    )
    parser.add_argument(
        "--hardware_unrolling",
        type=lambda x: tuple(map(int, x.split(','))),
        default=None,
        help="Hardware unroll dimensions for the accelerator."
    )
    parser.add_argument(
        "--dont_fuse_reshape",
        action="store_true",
        help="Whether to fuse reshape operations in the model."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Whether to run the pytorch evaluation during compilation"
    )
    add_qspec_args(parser)
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level))

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    quantizer = get_default_quantizer(
        input_activation=args.activation,
        output_activation=args.output_activation,
        weight=args.weight,
        bias=args.bias,
        force_scale_power_of_two=args.force_scale_power_of_two,
    )

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

    transform_args = {
        "patterns": vector_stages,
        "transpose_weight": args.transpose_weight,
        "transpose_fc": args.transpose_fc,
        "cache_size": args.cache_size,
        "num_banks": args.num_banks,
        "unroll_dims": args.hardware_unrolling,
        "conv2d_im2col": args.conv2d_im2col,
        "fuse_reshape": (
            not args.dont_fuse_reshape
            and (args.hardware_unrolling is None or max(args.hardware_unrolling) < 64)
        ),
    }

    compile_args = {
        "cache_size": args.cache_size,
        "num_banks": args.num_banks,
        "bank_width": args.bank_width,
        "unroll_dims": args.hardware_unrolling,
        "output_dir": args.model_output_dir,
        "output_file": args.model,
        "dump_verification_file": args.dump_verification_file,
    }

    if args.model in models.__dict__:
        model = torchvision_models.load_model(args)
        
        if args.dump_dataset or args.evaluate:
            imagenet_dataset = imagenet.retrieve_dataset(1000, "resnet")
            if args.evaluate:
                torchvision_models.evaluate(model, imagenet_dataset)
        else:
            imagenet_dataset = imagenet.retrieve_dataset(10, "resnet")

        gm, old_output, new_output, preprocess_fn = torchvision_models.quantize_and_dump_model(
            model=model,
            quantizer=quantizer,
            calibration_data=imagenet_dataset,
            vector_stages=vector_stages,
            args=args
        )

        if args.dump_dataset:
            preprocessed_imagenet = imagenet.dump_imagenet(
                args.dataset_output_dir, imagenet_dataset, "resnet", preprocess_fn, torch_dtype
            )
        
        if args.evaluate:
            torchvision_models.evaluate(gm, preprocessed_imagenet)
    elif args.model == "mobilebert":
        model, tokenizer = mobilebert.load_model(args)
        
        eval_dataset, train_dataset = glue.retrieve_dataset(model, tokenizer, args)
        
        if args.evaluate:
            mobilebert.evaluate(model, eval_dataset)

        if args.dump_dataset:
            preprocessed_dataset = glue.dump_dataset(
                args.dataset_output_dir, eval_dataset, model
            )

        gm, old_output, new_output = mobilebert.quantize_and_dump_model(
            model=model,
            quantizer=quantizer,
            calibration_data=train_dataset,
            vector_stages=vector_stages,
            args=args
        )

        if args.evaluate:
            mobilebert.evaluate_gm(gm, preprocessed_dataset)

    elif args.model == "bert":
        model, tokenizer = bert.load_model(args)

        eval_dataset, train_dataset = glue.retrieve_dataset(model, tokenizer, args)

        if args.evaluate:
            bert.evaluate(model, eval_dataset)

        if args.dump_dataset:
            preprocessed_dataset = glue.dump_dataset(
                args.dataset_output_dir, eval_dataset, model
            )

        gm, old_output, new_output = bert.quantize_and_dump_model(
            model=model,
            quantizer=quantizer,
            calibration_data=train_dataset,
            vector_stages=vector_stages,
            args=args
        )

        if args.evaluate:
            bert.evaluate_gm(gm, preprocessed_dataset)

    elif args.model == "llm_prefill" or args.model == "llm_decode":
        from transformers import AutoModelForCausalLM

        if args.model_name_or_path is None:
            args.model_name_or_path = "meta-llama/Llama-3.2-1B"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager", # turn off flash attention
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

        input_ids = encodings.input_ids[:,:args.context_length]

        past_key_values = None

        if args.model == "llm_decode":
            max_generated_length = input_ids.shape[1] + 128
            past_key_values = StaticCache(
                config=model.config,
                max_batch_size=1,
                max_cache_len=max_generated_length,
                device=model.device,
                dtype=model.dtype
            )

            with torch.no_grad():
                outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)

            input_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values

        inputs_embeds = model.model.embed_tokens(input_ids)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

        position_ids = cache_position.unsqueeze(0)

        causal_mask = model.model._update_causal_mask(
            None, inputs_embeds, cache_position, past_key_values, None
        )

        if args.model == "llm_prefill":
            causal_mask = causal_mask[:, :, :, : args.context_length]

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)

        example_args = (inputs_embeds, causal_mask, position_embeddings, cache_position)
        example_kwargs = {}

        class LlamaWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model.model
                self.lm_head = model.lm_head

                self.static_cache = past_key_values

                if self.static_cache is not None:
                    for i in range(len(self.static_cache.key_cache)):
                        self.register_buffer(f"key_cache_{i}", self.static_cache.key_cache[i], persistent=False)
                        self.register_buffer(f"value_cache_{i}", self.static_cache.value_cache[i], persistent=False)

            def forward(
                self,
                hidden_states,
                attention_mask,
                position_embeddings,
                cache_position=None,
            ):
                for decoder_layer in self.model.layers:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_embeddings=position_embeddings,
                        past_key_value=self.static_cache,
                        cache_position=cache_position,
                    )
                    hidden_states = layer_outputs[0]

                    if args.remove_duplicate:
                        break

                logits = self.lm_head(hidden_states)
                return logits

        if args.mixed_precision:
            set_qconfig(quantizer, get_llm_qscheme(args.hardware_unrolling[0], args.outlier_threshold))

        gm = prepare_pt2e(LlamaWrapper(), quantizer, example_args, example_kwargs)

        strip_softmax_dtype(gm)

        hidden_size = model.model.layers[0].input_layernorm.weight.shape[-1]
        example_input = torch.randn(1, 128, hidden_size, dtype=torch.bfloat16)
        replace_rmsnorm_with_layer_norm(gm, model.model.layers[0].input_layernorm, (example_input,))

        convert_pt2e(gm, args.bias)

        old_output = gm(*example_args, **example_kwargs)

        transform(gm, example_args, example_kwargs=example_kwargs, **transform_args)
        gm.graph.print_tabular()

        new_output = gm(*example_args, *list(example_kwargs.values()))

        compile(gm, example_args, **compile_args)
    elif args.model == "llm_kivi":
        from transformers import AutoModelForCausalLM

        if args.model_name_or_path is None:
            args.model_name_or_path = "meta-llama/Llama-3.2-1B"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            attn_implementation="eager", # turn off flash attention
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

        input_ids = encodings.input_ids[:,:args.context_length]

        max_length = args.context_length
        bs = 64 if args.hardware_unrolling is None else args.hardware_unrolling[0]

        swap_llama_attention(model)

        gm = convert_and_export_with_split_cache(
            model, max_len=max_length, max_new_tokens=bs
        ).module()

        # Run decode once to fill in the KV caches
        output = TorchExportableModuleWithStaticCache.generate(
            model,
            prompt_token_ids=input_ids,
            max_new_tokens=bs,
            min_length=max_length+1,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            model_decode=gm,
        )[0]

        remove_autocast_nodes(gm)
        strip_softmax_dtype(gm)

        quantizer.set_module_name_object_type_order(
            "model.model.rotary_emb", torch.ops.aten.matmul.default, 0, None
        )

        act0 = QuantizationSpec.from_str("int6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3")
        act1 = QuantizationSpec.from_str("int6,qs=microscaling,bs=64,ax=(-2,-1),scale=fp8_e5m3")
        matmul_qconfig = QuantizationConfig(act0, None, act1, None)

        for layer_idx in range(model.config.num_hidden_layers):
            module_name = (
                f"model.model.layers.slice(None, {model.config.num_hidden_layers}, None)"
                f"._modules.{layer_idx}.self_attn"
            )

            # Perform full cache matmul in MXINT6
            quantizer.set_module_name_object_type_order(
                module_name, torch.ops.aten.matmul.default, 0, matmul_qconfig
            )
            quantizer.set_module_name_object_type_order(
                module_name, torch.ops.aten.matmul.default, 2, matmul_qconfig
            )

            # Perform residual cache matmul in full precision
            quantizer.set_module_name_object_type_order(
                module_name, torch.ops.aten.matmul.default, 1, None
            )
            quantizer.set_module_name_object_type_order(
                module_name, torch.ops.aten.matmul.default, 3, None
            )

        from torch.ao.quantization.quantizer.utils import _annotate_output_qspec

        # if transpose key cache, quantize along the last axis.
        key_qspec = QuantizationSpec.from_str("uint2,bs=64,qs=group_wise_affine,ax=-1,scale=fp8_e5m3")
        value_qspec = QuantizationSpec.from_str("uint2,bs=64,qs=group_wise_affine,ax=-1,scale=fp8_e5m3")

        for node in gm.graph.nodes:
            match = re.match(r"^(key|value)_cache_(\d+)$", str(node.target))
            if node.op == "get_attr" and match is not None:
                _annotate_output_qspec(node, key_qspec if match.group(1) == "key" else value_qspec)

        gm = prepare_pt2e(gm, quantizer)

        hidden_size = model.model.layers[0].input_layernorm.weight.shape[-1]
        example_input = torch.randn(1, 1, hidden_size, dtype=torch.bfloat16)
        replace_rmsnorm_with_layer_norm(
            gm, model.model.layers[0].input_layernorm, (example_input,)
        )

        sink_obs_or_fq(gm)
        convert_pt2e(gm, eliminate_no_effect=False)

        example_input_ids = torch.tensor([[1]], dtype=torch.long)
        example_cache_position = torch.tensor([0], dtype=torch.long)
        example_cache_position_residual = torch.tensor([0], dtype=torch.long)
        example_attention_mask = torch.ones((1, max_length + bs), dtype=torch_dtype)[None, None, :, :]
        example_args = (
            example_input_ids,
            example_cache_position,
            example_cache_position_residual,
            example_attention_mask,
        )
        example_kwargs = {}

        old_output = gm(*example_args, **example_kwargs)

        fuse_dequantize_quantize(gm)

        transform(gm, example_args, example_kwargs, **transform_args)
        gm.graph.print_tabular()

        new_output = gm(*example_args, *list(example_kwargs.values()))

        compile(gm, example_args, **compile_args)
        gm.graph.print_tabular()
    elif args.model == "vit":
        model = vit.load_model(args) 

        if args.dump_dataset or args.evaluate:
            imagenet_dataset = imagenet.retrieve_dataset(1000, "vit")
            if args.evaluate:
                vit.evaluate(model, imagenet_dataset)
        else:
            imagenet_dataset = imagenet.retrieve_dataset(10, "vit")

        gm, old_output, new_output, preprocess_fn = vit.quantize_and_dump_model(
            model=model,
            quantizer=quantizer,
            calibration_data=imagenet_dataset,
            vector_stages=vector_stages,
            args=args
        )

        if args.dump_dataset:
            preprocessed_imagenet = imagenet.dump_imagenet(
                args.dataset_output_dir, imagenet_dataset, "vit", preprocess_fn, torch_dtype
            )

        if args.evaluate:
            vit.evaluate(gm, preprocessed_imagenet)

    elif args.model == "yolo5":
        import sys
        sys.path.append("libraries/yolov5-face")

        # Clear any previously loaded modules to avoid conflicts
        if 'utils' in sys.modules:
            del sys.modules['utils']

        from models.experimental import attempt_load

        model = attempt_load(args.model_name_or_path, map_location="cpu").eval()

        if args.bf16:
            model.bfloat16()

        example_args = (torch.randn(1, 3, 640, 640, dtype=torch_dtype),)
        output = model(*example_args)

        gm = prepare_pt2e(model, quantizer, example_args)

        from quantized_training.codegen.mapping import eliminate_dead_code
        eliminate_dead_code(gm.graph)

        dataset = load_dataset("CUHK-CSE/wider_face")

        pipeline = transforms.Compose([
            transforms.Resize((640, 640)),  # Resize to 416x416
            transforms.ToTensor()           # Convert to tensor and normalize to [0, 1]
        ])

        for i in tqdm(range(10)):
            inputs = pipeline(dataset['train'][i]["image"])
            with torch.no_grad():
                gm(inputs.unsqueeze(0).to(torch_dtype))

        convert_pt2e(gm, args.bias)

        old_output = gm(*example_args)[0]

        transform(gm, example_args, **transform_args)
        gm.graph.print_tabular()

        new_output = gm(*example_args)[0]

        compile(gm, example_args, **compile_args)
    elif args.model == "mobilevit":
        try:
            import timm
            from timm.layers import set_fused_attn
        except ImportError as e:
            raise ImportError("The 'timm' library is not installed. Please install it using 'pip install timm'.") from e

        set_fused_attn(False)
        model = timm.create_model("hf-hub:timm/mobilevit_xxs.cvnets_in1k", pretrained=True).eval()

        if args.bf16:
            model.bfloat16()

        example_args = (torch.randn(1, 3, 224, 224, dtype=torch_dtype),)
        gm = prepare_pt2e(model, quantizer, example_args)

        dataset = load_dataset("zh-plus/tiny-imagenet")

        image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

        for i in tqdm(range(10)):
            inputs = image_processor(dataset['train'][i]["image"], return_tensors="pt")
            with torch.no_grad():
                gm(inputs.pixel_values.to(torch_dtype))

        convert_pt2e(gm, args.bias)

        old_output = gm(*example_args)

        transform(gm, example_args, **transform_args, fuse_operator=False)

        gm, preprocess_fn = extract_input_preprocessor(gm)
        example_args = (preprocess_fn(example_args[0]),)

        fuse(gm, vector_stages, example_args)

        gm.graph.print_tabular()

        new_output = gm(*example_args)

        compile(gm, example_args, **compile_args)
    elif args.model == "mamba":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if args.model_name_or_path is None:
            args.model_name_or_path = "state-spaces/mamba-130m-hf"

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).eval()

        if args.bf16:
            model.bfloat16()

        input_ids = torch.randint(low=0, high=tokenizer.vocab_size, size=(1, 2))
        example_args = (input_ids,)
        example_kwargs = {"use_cache": False, "return_dict": False}

        gm = prepare_pt2e(model, quantizer, example_args, example_kwargs)

        convert_pt2e(gm, args.bias)

        old_output = gm(input_ids, False, False)[0]

        transform(gm, example_args, example_kwargs, patterns=vector_stages)
        gm.graph.print_tabular()

        new_output = gm(input_ids, False, False)[0]

        compile(gm, example_args, example_kwargs, **compile_args)
    else:
        raise ValueError(f"Model {args.model} not supported")

    try:
        assert torch.all(old_output == new_output)
        print("Results match")
    except Exception as e:
        print(e)
        print(old_output)
        print(new_output)
