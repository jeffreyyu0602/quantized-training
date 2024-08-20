import copy
import logging

import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
from torch.nn import Module
from torch.nn.utils.parametrize import type_before_parametrizations

from accelerate import dispatch_model
from transformers import PretrainedConfig

from quantized_training.modules import (
    Softmax,
    modeling_bert,
    modeling_mobilebert,
)
from quantized_training.qconfig import get_qconfig
from quantized_training.quantization_mappings import (
    DEFAULT_QAT_MODULE_MAPPINGS,
    QCONFIG_PROPAGATE_MODULE_CLASS_LIST,
    TRANSFORMER_MODULE_MAPPINGS,
)

__all__ = [
    "propagate_config",
    "quantize",
    "prepare",
    "convert",
    "replace_softmax",
    "get_quantized_model",
]

logger = logging.getLogger(__name__)

RESIDUAL_LAYERS_BWD = [
    "attention.self.query",
    "attention.self.key",
    "attention.self.value",
    "intermediate.dense",
    "bottleneck.input.dense",
    "bottleneck.attention.dense",
]

def propagate_config(module, name, qconfig):
    # TODO set qconfig ch_axis according to the module type and qscheme.
    setattr(module, name, qconfig)

    for child in module.children():
        propagate_config(child, name, qconfig)

def quantize(model, args, inplace=True):
    if not inplace:
        model = copy.deepcopy(model)

    if (
        args.activation is not None
        or args.error is not None
        or args.posit_exp
        or args.posit_exp_shifted
        or args.posit_reciprocal
    ) and (
        hasattr(model, 'config') and isinstance(model.config, PretrainedConfig)
    ):
        propagate_config(model, 'config', model.config)
        convert(model, inplace=True, custom_module_class_mapping=TRANSFORMER_MODULE_MAPPINGS)

    if hasattr(model, 'hf_device_map'):
        dispatch_model(model, device_map=model.hf_device_map)

    if args.posit_exp or args.posit_exp_shifted or args.posit_reciprocal:
        replace_softmax(
            model,
            posit_exp=args.posit_exp,
            posit_exp_shifted=args.posit_exp_shifted,
            posit_reciprocal=args.posit_reciprocal,
            dtype=torch.bfloat16 if args.bf16 else None,
        )

    if getattr(args, 'bf16', False):
        model.bfloat16()

    qconfig = get_qconfig(
        args.activation,
        args.weight,
        args.error,
        args.record_histogram,
        args.force_scale_power_of_two,
    )

    propagate_config(model, 'qconfig', qconfig)

    for name, param in model.named_parameters():
        if 'bias' in name:
            continue
        obs_or_fq = qconfig.weight(device=param.device)
        if getattr(obs_or_fq, 'qscheme', None) is not None:
            obs_or_fq(param.data)
        param.data = obs_or_fq(param.data)

    # If doing quantization aware training, swap QAT modules. LoRA has custom
    # implementation, so it needs to be swapped to match the training behavior.
    if args.weight is not None and (args.do_train or args.lora_rank > 0):
        convert(model, DEFAULT_QAT_MODULE_MAPPINGS, inplace=True)

    if args.activation is None:
        args.quantize_forward = None
    if args.error is None:
        args.quantize_backprop = None
    prepare(model, True, args.quantize_forward, args.quantize_backprop, args.op_fusion)

    return model

def _parse_ops(op_str):
    ops = {op.lower() for op in op_str.split(',')} if op_str is not None else set()
    valid_ops = set(QCONFIG_PROPAGATE_MODULE_CLASS_LIST.keys())
    invalid_ops = ops - valid_ops
    assert not invalid_ops, (
        f"Invalid operation(s) {', '.join(invalid_ops)}. Options are {', '.join(valid_ops)}."
    )
    return tuple(mod for op in ops for mod in QCONFIG_PROPAGATE_MODULE_CLASS_LIST[op])

def _get_unique_devices_(mod):
    return {p.device for p in mod.parameters()} | \
        {p.device for p in mod.buffers()}

def _register_module_hook(module, hook_name, name):
    obs_or_fq_dict = nn.ModuleDict()
    module.add_module(hook_name, obs_or_fq_dict)

    obs_or_fq_ctr = (
        module.qconfig.activation if hook_name == 'activation_pre_process'
        else module.qconfig.error
    )

    # TODO statically create all fake quant modules depends on the type of the
    # input module. Only MatmulFunctional, AddFunctional, and MulFunctional have
    # two inputs
    def observer_pre_hook(self, inputs):
        new_inputs = []
        for i, input in enumerate(inputs):
            if not isinstance(input, torch.Tensor):
                new_inputs.append(input)
                continue

            if (obs_or_fq_name := str(i)) not in obs_or_fq_dict:
                obs_or_fq = obs_or_fq_ctr(device=input.device)
                obs_or_fq.name = f"{name}.{obs_or_fq_name}"
                obs_or_fq_dict[obs_or_fq_name] = obs_or_fq
            new_inputs.append(obs_or_fq_dict[obs_or_fq_name](input))
        return tuple(new_inputs)

    def backward_hook(self, grad_inputs, grad_outputs):
        return observer_pre_hook(self, grad_inputs)

    if hook_name == 'activation_pre_process':
        module.register_forward_pre_hook(observer_pre_hook)
    elif hook_name == 'error_pre_process':
        module.register_full_backward_pre_hook(observer_pre_hook)
    elif hook_name == 'error_post_process':
        module.register_full_backward_hook(backward_hook)

def _add_observer_(
        module, fwd_pre_hook_module_list, bwd_pre_hook_module_list,
        bwd_residual, op_fusion, prefix):
    def insert_obs_or_fq(m, name):
        if not hasattr(m, 'qconfig') or m.qconfig is None:
            return
        if op_fusion is not None and any(layer in name for layer in op_fusion):
            return
        if isinstance(m, fwd_pre_hook_module_list):
            _register_module_hook(m, 'activation_pre_process', name)
        if isinstance(m, bwd_pre_hook_module_list):
            _register_module_hook(m, 'error_pre_process', name)
        if bwd_residual and (
            any(layer in name for layer in RESIDUAL_LAYERS_BWD)
            or isinstance(m, _parse_ops("residual"))
        ):
            _register_module_hook(m, 'error_post_process', name)

    named_modules = dict(module.named_children())
    for name, child in named_modules.items():
        module_prefix = prefix + '.' + name if prefix else name
        if isinstance(child, nni._FusedModule):
            insert_obs_or_fq(child, module_prefix)
        else:
            _add_observer_(
                child, fwd_pre_hook_module_list, bwd_pre_hook_module_list,
                bwd_residual, op_fusion, module_prefix)
    insert_obs_or_fq(module, prefix)

def prepare(
        model, inplace=False, fwd_quantized_ops=None, bwd_quantized_ops=None,
        op_fusion=None):
    if not inplace:
        model = copy.deepcopy(model)

    fwd_pre_hook_module_list = _parse_ops(fwd_quantized_ops)
    bwd_pre_hook_module_list = _parse_ops(bwd_quantized_ops)
    is_bwd_residual = bwd_quantized_ops and "residual" in bwd_quantized_ops
    _add_observer_(
        model, fwd_pre_hook_module_list, bwd_pre_hook_module_list,
        is_bwd_residual, op_fusion, prefix='')
    return model

def convert(module, mapping=None, inplace=False, custom_module_class_mapping=None):
    if not inplace:
        module = copy.deepcopy(module)
    _convert(
        module, mapping, inplace=True,
        custom_module_class_mapping=custom_module_class_mapping)
    return module

def _convert(module, mapping=None, inplace=False, custom_module_class_mapping=None):
    r"""Converts submodules in input mod to a different mod according to `mapping`
    by calling `from_float` method on the target mod class

    Args:
        mod: input mod
        mapping: a dictionary that maps from source mod type to target
                 mod type, can be overwritten to allow swapping user defined
                 Modules
        inplace: carry out model transformations in-place, the original mod
                 is mutated

    """
    if mapping is None:
        mapping = DEFAULT_QAT_MODULE_MAPPINGS
    if custom_module_class_mapping is None:
        custom_module_class_mapping = {}

    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}
    for name, mod in module.named_children():
        # both fused modules and observed custom modules are
        # swapped as one unit
        if (not isinstance(mod, nni._FusedModule)
            and type_before_parametrizations(mod) not in custom_module_class_mapping):
            _convert(mod, mapping, True, custom_module_class_mapping)
        reassign[name] = swap_module(mod, mapping, custom_module_class_mapping)

    for key, value in reassign.items():
        module._modules[key] = value

    return module

def swap_module(mod, mapping, custom_module_class_mapping):
    r"""Swaps the mod if it has a quantized counterpart and it has an
    `observer` attached.

    Args:
        mod: input mod
        mapping: a dictionary that maps from nn mod to nnq mod

    Return:
        The corresponding quantized mod of `mod`
    """
    new_mod = mod
    swapped = False
    if type_before_parametrizations(mod) in custom_module_class_mapping:
        new_mod = custom_module_class_mapping[type_before_parametrizations(mod)].from_observed(mod)
        swapped = True
    elif (
        hasattr(mod, 'qconfig')
        and mod.qconfig is not None
        and type_before_parametrizations(mod) in mapping
    ):
        new_mod = mapping[type_before_parametrizations(mod)].from_float(mod)
        swapped = True

    if swapped:
        # Preserve module's pre forward hooks. They'll be called on quantized input
        for pre_hook_fn in mod._forward_pre_hooks.values():
            new_mod.register_forward_pre_hook(pre_hook_fn)
        # Preserve module's forward hooks. They'll be called on inputs to residual
        for hook_fn in mod._forward_hooks.values():
            new_mod.register_forward_hook(hook_fn)
        # Preserve module's pre backward hooks. They'll be called on input gradients
        for pre_hook_fn in mod._backward_pre_hooks.values():
            new_mod.register_full_backward_pre_hook(pre_hook_fn)
        # Preserve module's backward hooks. They'll be called on input gradients to residual
        for hook_fn in mod._backward_hooks.values():
            new_mod.register_full_backward_hook(hook_fn)

        # respect device affinity when swapping modules
        devices = _get_unique_devices_(mod)
        assert len(devices) <= 1, (
            f"swap_module only works with cpu or single-device CUDA modules, but got devices {devices}"
        )
        device = next(iter(devices)) if len(devices) > 0 else None
        if device:
            new_mod.to(device)
    return new_mod

def replace_softmax(
    module: Module,
    posit_exp: bool,
    posit_exp_shifted: bool,
    posit_reciprocal: bool,
    dtype=None,
    device=None
):
    if device is None:
        devices = _get_unique_devices_(module)
        device = next(iter(devices)) if len(devices) == 1 else None

    for name, mod in module.named_children():
        if type_before_parametrizations(mod) == nn.Softmax:
            new_mod = Softmax(posit_exp, posit_exp_shifted, posit_reciprocal,
                              dim=-1, dtype=dtype, device=device)
            setattr(module, name, new_mod)
        else:
            replace_softmax(mod, posit_exp, posit_exp_shifted, posit_reciprocal, dtype, device)

def get_quantized_model(model, qconfig, op_fusion=None, device=None):
    logger.info(f"Fusing operations: {op_fusion}")

    if device is None:
        devices = _get_unique_devices_(model)
        assert len(devices) <= 1, (
            f"Quantized model only works with cpu or single-device CUDA modules, but got devices {devices}"
        )
        device = next(iter(devices)) if len(devices) > 0 else None

    act_fake_quant = qconfig.activation(device=device)
    class Quantizer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, layer=None):
            ctx.layer = layer
            if op_fusion and any(x in layer for x in op_fusion):
                return input
            return act_fake_quant(input)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None

    model_name = type(model).__name__
    model_type = model_name.split("For", 1)[0]
    assert model_type in {"MobileBert", "Bert"}, (
        f"'{model_type}' models are not support for quantization."
    )

    module = modeling_bert if model_type == "Bert" else modeling_mobilebert
    quantized_model = getattr(module, model_name)(model.config, Quantizer.apply)
    quantized_model.load_state_dict(model.state_dict())
    quantized_model.to(device)

    return quantized_model
