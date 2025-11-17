# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert ViT and non-distilled DeiT checkpoints from the timm library."""

import requests
import torch
from PIL import Image
from tqdm import tqdm

from transformers import DeiTImageProcessor, ViTConfig, ViTForImageClassification, ViTImageProcessor, ViTModel
from transformers.utils import logging

from quantized_training import (
    DerivedQuantizationSpec,
    FusedAmaxObsFakeQuantize,
    QuantizationConfig,
    QuantizationSpec,
    add_qspec_args,
    convert_pt2e,
    export_model,
    prepare_pt2e,
    transform,
    compile,
    derive_bias_qparams_fn,
    extract_input_preprocessor,
    fuse,
)
from quantized_training.codegen import (
    get_conv_bn_layers,
    pad_vit_embeddings_output,
    remove_softmax_dtype_cast,
)

from .utils import get_compile_args, get_transform_args

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, base_model=False):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append((f"blocks.{i}.norm1.weight", f"vit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"vit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"vit.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"vit.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"vit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"vit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"vit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"vit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"vit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"vit.encoder.layer.{i}.output.dense.bias"))

    # projection layer + position embeddings
    rename_keys.extend(
        [
            ("cls_token", "vit.embeddings.cls_token"),
            ("patch_embed.proj.weight", "vit.embeddings.patch_embeddings.projection.weight"),
            ("patch_embed.proj.bias", "vit.embeddings.patch_embeddings.projection.bias"),
            ("pos_embed", "vit.embeddings.position_embeddings"),
        ]
    )

    if base_model:
        # layernorm
        rename_keys.extend(
            [
                ("norm.weight", "layernorm.weight"),
                ("norm.bias", "layernorm.bias"),
            ]
        )

        # if just the base model, we should remove "vit" from all keys that start with "vit"
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("vit") else pair for pair in rename_keys]
    else:
        # layernorm + classification head
        rename_keys.extend(
            [
                ("norm.weight", "vit.layernorm.weight"),
                ("norm.bias", "vit.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config, base_model=False):
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ""
        else:
            prefix = "vit."
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_vit_checkpoint(vit_name):
    """
    Copy/paste/tweak model's weights to our ViT structure.
    """
    try:
        import timm
        from timm.data import ImageNetInfo, infer_imagenet_subset
    except ImportError:
        raise ImportError(
            "Please install the timm library to convert ViT checkpoints from timm: `pip install timm`."
        )

    # define default ViT configuration
    config = ViTConfig()
    base_model = False

    # load original model from timm
    timm_model = timm.create_model(vit_name, pretrained=True)
    timm_model.eval()

    # detect unsupported ViT models in transformers
    # fc_norm is present
    if not isinstance(getattr(timm_model, "fc_norm", None), torch.nn.Identity):
        raise ValueError(f"{vit_name} is not supported in transformers because of the presence of fc_norm.")

    # use of global average pooling in combination (or without) class token
    if getattr(timm_model, "global_pool", None) == "avg":
        raise ValueError(f"{vit_name} is not supported in transformers because of use of global average pooling.")

    # CLIP style vit with norm_pre layer present
    if "clip" in vit_name and not isinstance(getattr(timm_model, "norm_pre", None), torch.nn.Identity):
        raise ValueError(
            f"{vit_name} is not supported in transformers because it's a CLIP style ViT with norm_pre layer."
        )

    # SigLIP style vit with attn_pool layer present
    if "siglip" in vit_name and getattr(timm_model, "global_pool", None) == "map":
        raise ValueError(
            f"{vit_name} is not supported in transformers because it's a SigLIP style ViT with attn_pool."
        )

    # use of layer scale in ViT model blocks
    if not isinstance(getattr(timm_model.blocks[0], "ls1", None), torch.nn.Identity) or not isinstance(
        getattr(timm_model.blocks[0], "ls2", None), torch.nn.Identity
    ):
        raise ValueError(f"{vit_name} is not supported in transformers because it uses a layer scale in its blocks.")

    # Hybrid ResNet-ViTs
    if not isinstance(timm_model.patch_embed, timm.layers.PatchEmbed):
        raise ValueError(f"{vit_name} is not supported in transformers because it is a hybrid ResNet-ViT.")

    # get patch size and image size from the patch embedding submodule
    config.patch_size = timm_model.patch_embed.patch_size[0]
    config.image_size = timm_model.patch_embed.img_size[0]

    # retrieve architecture-specific parameters from the timm model
    config.hidden_size = timm_model.embed_dim
    config.intermediate_size = timm_model.blocks[0].mlp.fc1.out_features
    config.num_hidden_layers = len(timm_model.blocks)
    config.num_attention_heads = timm_model.blocks[0].attn.num_heads

    # check whether the model has a classification head or not
    if timm_model.num_classes != 0:
        config.num_labels = timm_model.num_classes
        # infer ImageNet subset from timm model
        imagenet_subset = infer_imagenet_subset(timm_model)
        dataset_info = ImageNetInfo(imagenet_subset)
        config.id2label = {i: dataset_info.index_to_label_name(i) for i in range(dataset_info.num_classes())}
        config.label2id = {v: k for k, v in config.id2label.items()}
    else:
        print(f"{vit_name} is going to be converted as a feature extractor only.")
        base_model = True

    # load state_dict of original model
    state_dict = timm_model.state_dict()

    # remove and rename some keys in the state dict
    if base_model:
        remove_classification_head_(state_dict)
    rename_keys = create_rename_keys(config, base_model)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, base_model)

    # load HuggingFace model
    if base_model:
        model = ViTModel(config, add_pooling_layer=False).eval()
    else:
        model = ViTForImageClassification(config).eval()
    model.load_state_dict(state_dict)

    # Check outputs on an image, prepared by ViTImageProcessor/DeiTImageProcessor
    if "deit" in vit_name:
        image_processor = DeiTImageProcessor(size=config.image_size)
    else:
        image_processor = ViTImageProcessor(size=config.image_size)
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values)

    if base_model:
        timm_pooled_output = timm_model.forward_features(pixel_values)
        assert timm_pooled_output.shape == outputs.last_hidden_state.shape
        assert torch.allclose(timm_pooled_output, outputs.last_hidden_state, atol=1e-1)
    else:
        timm_logits = timm_model(pixel_values)
        assert timm_logits.shape == outputs.logits.shape
        assert torch.allclose(timm_logits, outputs.logits, atol=1e-3)
    return model


def load_model(args):
    from transformers import ViTForImageClassification

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    model_name_or_path = None

    # for timm models, it needs to be converted to pytorch first
    if args.model_name_or_path is None or "timm" in args.model_name_or_path:
        model_name_or_path = "google/vit-base-patch16-224"
    else:
        model_name_or_path = args.model_name_or_path

    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
    )

    if args.model_name_or_path is not None and "timm" in args.model_name_or_path:
        timm_model = convert_vit_checkpoint(args.model_name_or_path)
        model.load_state_dict(timm_model.state_dict(), strict=False)
    return model

def quantize_and_dump_model(model, quantizer, calibration_data, vector_stages, args):
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    transform_args = get_transform_args(args, vector_stages)
    compile_args = get_compile_args(args)

    modules_to_fuse = get_conv_bn_layers(model)
    if len(modules_to_fuse) > 0:
        model = torch.ao.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

    quantizer.set_module_name("classifier", None)

    if args.activation is not None and "microscaling" in args.activation:
        dtype = args.activation.split(",")[0]
        if dtype == "nf4_6":
            dtype = "int6"
        qspec = QuantizationSpec.from_str(f"{dtype},qs=per_tensor_symmetric")
        qspec.observer_or_fake_quant_ctr = FusedAmaxObsFakeQuantize

        bias_qspec = DerivedQuantizationSpec(
            derived_from=None,
            derive_qparams_fn=derive_bias_qparams_fn,
            dtype=None,
        )

        qconfig = QuantizationConfig(qspec, None, qspec, bias_qspec)
        quantizer.set_module_name("^vit.embeddings.patch_embeddings.projection$", qconfig)

    example_args = (calibration_data[0]["image"].to(torch_dtype),)

    gm = export_model(model, example_args)
    pad_vit_embeddings_output(
        gm,
        model.vit.embeddings,
        example_args,
        unroll=args.hardware_unrolling[1]
    )

    gm = prepare_pt2e(gm, quantizer)

    remove_softmax_dtype_cast(gm)

    for i in tqdm(range(10), desc="Calibrating ViT"):
        inputs = calibration_data[i]["image"]
        with torch.no_grad():
            gm(inputs.to(torch_dtype))

    convert_pt2e(gm, args.bias)

    old_output = gm(*example_args).logits

    transform(gm, example_args, **transform_args, fuse_operator=False)

    gm, preprocess_fn = extract_input_preprocessor(gm)
    example_args = (preprocess_fn(example_args[0]),)

    fuse(gm, vector_stages, example_args)

    gm.graph.print_tabular()
    new_output = gm(*example_args).logits

    compile(gm, example_args, **compile_args)
    return gm, old_output, new_output, preprocess_fn

def evaluate(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for image_label_pair in tqdm(dataset, desc="Evaluating ViT"):
            # for running the original model without the preprocessing function 
            # applied to the dataset
            image = image_label_pair["image"].to(device)
            label = image_label_pair["label"]
    
            outputs = model(image)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1)
            if prediction.item() == label:
                correct_predictions += 1
            total_samples += 1

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"Vit Accuracy: {accuracy:.4f}")

    
            

