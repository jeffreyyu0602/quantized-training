import argparse
import logging
import os
import sys

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSemanticSegmentation,
    AutoImageProcessor,
    AutoTokenizer,
    StaticCache,
    default_data_collator,
)
from tqdm import tqdm

from quantized_training import (
    DerivedQuantizationSpec,
    FusedAmaxObsFakeQuantize,
    QuantizationConfig,
    QuantizationSpec,
    add_qspec_args,
    convert_pt2e,
    get_default_quantizer,
    prepare_pt2e,
    transform,
    compile,
    derive_bias_qparams_fn
)
from quantized_training.codegen.utils import (
    get_conv_bn_layers,
    pad_vit_embeddings_output,
    replace_interpolate,
    replace_rmsnorm_with_layer_norm,
    strip_softmax_dtype,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(script_dir, '../examples/language_modeling')
sys.path.append(os.path.abspath(target_path))

from prepare_model import set_qscheme

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


TORCHVISION_MODELS = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "mobilenet": models.mobilenet_v2,
}


vector_stages = [
    [
        ["gemm"],
        ["dequantize"],
        ["add", "sub", "mul", "div"],
        ["exp", "abs", "relu", "gelu", "silu", "vmap"],
        ["add", "mul", "div"],
        ["div", "quantize"],
    ],
    [
        ["layer_norm", torch.nn.Softmax, torch.nn.functional.softmax],
        ["quantize"],
    ]
]


LLAMA_MP_QSCHEME = {
    r"self_attn\.q_proj$": [
        "int2,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
        "int2,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
    ],
    r"self_attn\.k_proj$": [
        "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
        "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
    ],
    r"self_attn\.v_proj$": [
        "int6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
        "int2,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
    ],
    r"self_attn\.o_proj$": [
        "int6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
        "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
    ],
    torch.ops.aten.matmul.default: [
        "int6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
        "int6,qs=microscaling,bs=64,ax=-2,scale=fp8_e5m3",
    ],
}


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=10)
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
        "--output_dir",
        required=True,
        help="Output directory for generated tensor files"
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
    add_qspec_args(parser)
    args = parser.parse_args()

    quantizer = get_default_quantizer(
        input_activation=args.activation,
        output_activation=args.output_activation,
        weight=args.weight,
        bias=args.bias,
        force_scale_power_of_two=args.force_scale_power_of_two,
    )

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

    compile_args = {
        "bank_width": args.bank_width,
        "output_dir": args.output_dir,
        "output_file": args.model,
    }

    if args.model in TORCHVISION_MODELS:
        model = TORCHVISION_MODELS[args.model](pretrained=True).eval()

        if args.model_name_or_path:
            checkpoint = torch.load(args.model_name_or_path, map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'], strict=False)

        if args.bf16:
            model.bfloat16()

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
        gm = prepare_pt2e(model, quantizer, example_args)

        dataset = load_dataset("zh-plus/tiny-imagenet")

        image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

        for i in tqdm(range(10)):
            inputs = image_processor(dataset['train'][i]["image"], return_tensors="pt")
            with torch.no_grad():
                gm(inputs.pixel_values.to(torch_dtype))

        convert_pt2e(gm, args.bias)

        orig_output, new_output = transform(gm, example_args, patterns=vector_stages)
        compile(gm, example_args, **compile_args)
    elif args.model == "segformer":
        replace_interpolate()

        if args.model_name_or_path is None:
            args.model_name_or_path = "nvidia/segformer-b0-finetuned-ade-512-512"

        model = AutoModelForSemanticSegmentation.from_pretrained(args.model_name_or_path).eval()

        modules_to_fuse = ["decode_head.linear_fuse", "decode_head.batch_norm"]
        model = torch.ao.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

        if args.bf16:
            model.bfloat16()

        dataset = load_dataset("zh-plus/tiny-imagenet")

        import torchvision.transforms as transforms
        preprocess = transforms.Compose([
            transforms.RandomResizedCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        inputs = preprocess(dataset['train'][0]["image"])
        example_args = (inputs.unsqueeze(0).to(torch_dtype),)
        gm = prepare_pt2e(model, quantizer, example_args)

        for i in tqdm(range(10)):
            inputs = preprocess(dataset['train'][i]["image"])
            with torch.no_grad():
                gm(inputs.unsqueeze(0).to(torch_dtype))

        convert_pt2e(gm, args.bias)

        # TODO why the output is different after replacing gelu with vmap
        orig_output, new_output = transform(gm, example_args, patterns=vector_stages)
        compile(gm, example_args, **compile_args)

        orig_output = orig_output.logits
        new_output = new_output.logits
    elif args.model == "mobilebert":
        if args.model_name_or_path is None:
            args.model_name_or_path = "google/mobilebert-uncased"

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            attn_implementation="eager",
        ).eval()

        if args.bf16:
            model.bfloat16()

        # Setup SST-2 dataset
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

        batch = next(iter(train_dataloader))
        input_ids = batch["input_ids"]
        input_shape = input_ids.size()

        embedding_output = model.mobilebert.embeddings(
            input_ids=input_ids,
            token_type_ids=batch["token_type_ids"]
        )

        extended_attention_mask = model.mobilebert.get_extended_attention_mask(batch["attention_mask"], input_shape)

        head_mask = model.mobilebert.get_head_mask(None, model.config.num_hidden_layers)

        example_args = (embedding_output, extended_attention_mask, head_mask)

        class MobileBertWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mobilebert = model.mobilebert
                self.classifier = model.classifier

            def forward(self, hidden_states, attention_mask, head_mask):
                for i, layer_module in enumerate(self.mobilebert.encoder.layer):
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask=attention_mask,
                        head_mask=head_mask[i],
                    )
                    hidden_states = layer_outputs[0]

                    if args.remove_duplicate:
                        break

                first_token_tensor = hidden_states[:, 0]
                output = self.classifier(first_token_tensor)
                return output

        quantizer.set_module_name("classifier", None)

        gm = prepare_pt2e(MobileBertWrapper(), quantizer, example_args)

        for step, batch in enumerate(tqdm(train_dataloader)):
            embedding_output = model.mobilebert.embeddings(
                input_ids=batch["input_ids"],
                token_type_ids=batch["token_type_ids"]
            )
            gm(embedding_output, extended_attention_mask, head_mask)

            if step == args.calibration_steps:
                break

        convert_pt2e(gm, args.bias)

        orig_output, new_output = transform(gm, example_args, patterns=vector_stages)
        compile(gm, example_args, **compile_args)
    elif args.model == "bert":
        if args.model_name_or_path is None:
            args.model_name_or_path = "bert-base-uncased"

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            attn_implementation="eager",
        ).eval()

        if args.bf16:
            model.bfloat16()

        input_ids = torch.randint(0, 30522, (1, 128), dtype=torch.long)
        input_shape = input_ids.size()

        token_type_ids = torch.zeros(input_shape, dtype=torch.long)
        position_ids = torch.ones(input_shape, dtype=torch.long)
        head_mask = None

        embedding_output = model.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        attention_mask = torch.ones(input_shape)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = model.bert.get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = model.bert.get_head_mask(head_mask, model.config.num_hidden_layers)

        class BertWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = model.bert
                self.classifier = model.classifier

            def forward(self, hidden_states, attention_mask, head_mask):
                for i, layer_module in enumerate(self.bert.encoder.layer):
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask=attention_mask,
                        head_mask=head_mask[i],
                    )
                    hidden_states = layer_outputs[0]

                    if args.remove_duplicate:
                        break

                first_token_tensor = hidden_states[:, 0]
                output = self.classifier(first_token_tensor)
                return output

        example_args = (embedding_output, extended_attention_mask, head_mask)

        quantizer.set_module_name("classifier", None)

        gm = prepare_pt2e(BertWrapper(), quantizer, example_args)
        convert_pt2e(gm, args.bias)

        orig_output, new_output = transform(gm, example_args, patterns=vector_stages)
        compile(gm, example_args, **compile_args)
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
            max_generated_length = input_ids.shape[1] + 64
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
            set_qscheme(quantizer, LLAMA_MP_QSCHEME)

        gm = prepare_pt2e(LlamaWrapper(), quantizer, example_args, example_kwargs)

        strip_softmax_dtype(gm)

        hidden_size = model.model.layers[0].input_layernorm.weight.shape[-1]
        example_input = torch.randn(1, 128, hidden_size, dtype=torch.bfloat16)
        replace_rmsnorm_with_layer_norm(gm, model.model.layers[0].input_layernorm, (example_input,))

        convert_pt2e(gm, args.bias)

        orig_output, new_output = transform(
            gm, example_args, example_kwargs=example_kwargs, patterns=vector_stages
        )

        compile(gm, example_args, **compile_args)
    elif args.model == "vit":
        from transformers import ViTForImageClassification

        if args.model_name_or_path is None:
            args.model_name_or_path = "google/vit-base-patch16-224"

        model = ViTForImageClassification.from_pretrained(
            args.model_name_or_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16 if args.bf16 else None,
        )

        modules_to_fuse = get_conv_bn_layers(model)
        if len(modules_to_fuse) > 0:
            model = torch.ao.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

        quantizer.set_module_name("classifier", None)

        dataset = load_dataset("zh-plus/tiny-imagenet")

        image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

        inputs = image_processor(dataset['train'][0]["image"], return_tensors="pt")
        example_args = (inputs.pixel_values.to(torch_dtype),)
        gm = prepare_pt2e(model, quantizer, example_args)

        strip_softmax_dtype(gm)

        for i in tqdm(range(10)):
            inputs = image_processor(dataset['train'][i]["image"], return_tensors="pt")
            with torch.no_grad():
                gm(inputs.pixel_values.to(torch_dtype))

        convert_pt2e(gm, args.bias)

        pad_vit_embeddings_output(gm, model.vit.embeddings, example_args)

        orig_output, new_output = transform(gm, example_args, patterns=vector_stages)
        compile(gm, example_args, **compile_args)

        orig_output = orig_output.logits
        new_output = new_output.logits
    elif args.model == "yolo5":
        import sys
        sys.path.append("libraries/yolov5-face")

        from models.experimental import attempt_load

        model = attempt_load(args.model_name_or_path, map_location="cpu").eval()

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

        orig_output, new_output = transform(gm, example_args, patterns=vector_stages)
        compile(gm, example_args, **compile_args)

        orig_output = orig_output[0]
        new_output = new_output[0]
    elif args.model == "mobilevit":
        try:
            import timm
            from timm.layers import set_fused_attn
        except ImportError as e:
            raise ImportError("The 'timm' library is not installed. Please install it using 'pip install timm'.") from e

        set_fused_attn(False)
        model = timm.create_model("hf-hub:timm/mobilevit_xxs.cvnets_in1k", pretrained=True).eval()

        example_args = (torch.randn(1, 3, 224, 224, dtype=torch_dtype),)
        gm = prepare_pt2e(model, quantizer, example_args)

        dataset = load_dataset("zh-plus/tiny-imagenet")

        image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

        for i in tqdm(range(10)):
            inputs = image_processor(dataset['train'][i]["image"], return_tensors="pt")
            with torch.no_grad():
                gm(inputs.pixel_values.to(torch_dtype))

        convert_pt2e(gm, args.bias)

        orig_output, new_output = transform(gm, example_args, patterns=vector_stages)
        compile(gm, example_args, **compile_args)
    else:
        raise ValueError(f"Model {args.model} not supported")

    try:
        assert torch.all(orig_output == new_output)
        print("Results match")
    except Exception as e:
        print(e)
        print(orig_output)
        print(new_output)
