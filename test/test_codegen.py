import argparse
import logging
from typing import List

import torch
from datasets import load_dataset
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
    QuantizationConfig,
    QuantizationSpec,
    add_qspec_args,
    convert_pt2e,
    get_default_quantizer,
    prepare_pt2e,
    transform,
)
from quantized_training.codegen.utils import (
    eliminate_dtype_conversion,
    get_conv_bn_layers,
    pad_vit_embeddings_output,
    replace_interpolate,
    replace_rmsnorm_with_layer_norm,
)

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

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

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

        example_args = (torch.randn(1, 3, 224, 224, dtype=torch_dtype),)
        gm = prepare_pt2e(model, quantizer, example_args)

        dataset = load_dataset("zh-plus/tiny-imagenet")

        image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

        for i in tqdm(range(10)):
            inputs = image_processor(dataset['train'][i]["image"], return_tensors="pt")
            with torch.no_grad():
                gm(inputs.pixel_values.to(torch_dtype))

        convert_pt2e(gm)

        orig_output, new_output = transform(
            gm,
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

        convert_pt2e(gm)

        # TODO why the output is different after replacing gelu with vmap
        orig_output, new_output = transform(
            gm,
            example_args,
            output_file="segformer",
            output_dir=args.output_dir,
        )
        orig_output = orig_output.logits
        new_output = new_output.logits
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

        orig_output, new_output = transform(
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

        for i in range(3):
            gm(*example_args)

        for name, module in gm.named_modules():
            if hasattr(module, "scale"):
                print(module.scale)

        convert_pt2e(gm, args.bias)

        orig_output, new_output = transform(
            gm,
            example_args,
            output_file="mobilebert",
            output_dir=args.output_dir,
        )

        orig_output = orig_output[0]
        new_output = new_output[0]
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

        quantizer.set_module_name("classifier", None)

        gm = prepare_pt2e(BertNoEmbed(), quantizer, example_args)
        convert_pt2e(gm)

        orig_output, new_output = transform(
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

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

        input_ids = encodings.input_ids[:,:128]
        inputs_embeds = model.model.embed_tokens(input_ids)
        cache_position = torch.arange(0, inputs_embeds.shape[1])
        position_ids = cache_position.unsqueeze(0)
        causal_mask = model.model._update_causal_mask(
            None, inputs_embeds, cache_position, None, None
        )

        # create position embeddings to be shared across the decoder layers
        position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)

        example_args = (inputs_embeds, causal_mask, position_embeddings)

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

        hidden_size = model.model.layers[0].input_layernorm.weight.shape[-1]
        example_input = torch.randn(1, 128, hidden_size, dtype=torch.bfloat16)
        replace_rmsnorm_with_layer_norm(gm, model.model.layers[0].input_layernorm, (example_input,))

        eliminate_dtype_conversion(gm)

        convert_pt2e(gm)

        orig_output, new_output = transform(
            gm,
            example_args,
            output_dir=args.output_dir,
        )
    elif args.model == "llama_decoder":
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

        input_ids = encodings.input_ids[:,:128]
        inputs_embeds = model.model.embed_tokens(input_ids)
        cache_position = torch.arange(0, inputs_embeds.shape[1])
        position_ids = cache_position.unsqueeze(0)
        causal_mask = model.model._update_causal_mask(
            None, inputs_embeds, cache_position, None, None
        )

        # no matter the length, we just slice it
        causal_mask = causal_mask[:, :, :, : input_ids.shape[-1]]

        # create position embeddings to be shared across the decoder layers
        position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)

        class LLamaDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model.model

            def forward(self, hidden_states, causal_mask, position_embeddings):
                layer_outputs = self.model.layers[0](
                    hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                )
                return layer_outputs[0]

        example_args = (inputs_embeds, causal_mask, position_embeddings)
        model = LLamaDecoder()

        gm = prepare_pt2e(model, quantizer, example_args)

        # Calibrate using random inputs
        for i in range(3):
            calib_input = (inputs_embeds.clone(), causal_mask, position_embeddings)
            gm(*calib_input)

        hidden_size = model.model.layers[0].input_layernorm.weight.shape[-1]
        example_input = torch.randn(1, 128, hidden_size, dtype=torch.bfloat16)
        replace_rmsnorm_with_layer_norm(gm, model.model.layers[0].input_layernorm, (example_input,))

        eliminate_dtype_conversion(gm)

        convert_pt2e(gm)

        # Generate float32 model
        if not args.bf16:
            gm.float()
            position_embeddings = tuple(t.float() for t in position_embeddings)
            example_args = (inputs_embeds.float(), causal_mask.float(), position_embeddings)

        orig_output, new_output = transform(
            gm,
            example_args,
            output_dir=args.output_dir,
        )
    elif args.model == "llama_decode":
        import transformers

        model_id = "meta-llama/Meta-Llama-3-8B"

        pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
        pipeline("Hey how are you doing today?")
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

        for i in tqdm(range(10)):
            inputs = image_processor(dataset['train'][i]["image"], return_tensors="pt")
            with torch.no_grad():
                gm(inputs.pixel_values.to(torch_dtype))

        convert_pt2e(gm)

        pad_vit_embeddings_output(gm, model.vit.embeddings, example_args)

        orig_output, new_output = transform(
            gm,
            example_args,
            output_file=args.model,
            output_dir=args.output_dir,
        )

        orig_output = orig_output.logits
        new_output = new_output.logits
    else:
        raise ValueError(f"Model {args.model} not supported")

    try:
        assert torch.all(orig_output == new_output)
        print("Results match")
    except Exception as e:
        print(e)
        print(orig_output)
        print(new_output)
