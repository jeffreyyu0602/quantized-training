import copy
import logging

import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
from torch.nn import Module
from torch.nn.utils.parametrize import type_before_parametrizations

from accelerate import dispatch_model
from transformers import PretrainedConfig

from .fake_quantize import FusedAmaxObsFakeQuantize
from .modules import Softmax, modeling_bert, modeling_mobilebert
from .qconfig import QConfig
from .quantization_mappings import (
    QCONFIG_PROPAGATE_MODULE_CLASS_LIST,
    DEFAULT_QAT_MODULE_MAPPINGS,
    HF_TRANSFORMER_MODULE_MAPPINGS,
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

RESIDUAL_LAYERS = [
    "query", "key", "value", "intermediate", "bottleneck.input", "bottleneck.attention",
]

def propagate_config(module, name, qconfig):
    setattr(module, name, qconfig)

    for child in module.children():
        propagate_config(child, name, qconfig)

def _create_fake_quant(qconfig, args):
    if qconfig is None:
        return nn.Identity()

    return FusedAmaxObsFakeQuantize.with_args(
        **qconfig.to_dict(), record_histogram=args.record_histogram
    )

def quantize(model, args, run_fn=None, inplace=True):
    if not inplace:
        model = copy.deepcopy(model)

    act_fake_quant = _create_fake_quant(args.activation, args)
    wt_fake_quant = _create_fake_quant(args.weight, args)
    error_fake_quant = _create_fake_quant(args.error, args)
    qconfig = QConfig(
        activation=act_fake_quant, weight=wt_fake_quant, error=error_fake_quant
    )

    if (
        (args.activation or args.error or args.posit_exp or
            args.posit_exp_shifted or args.posit_reciprocal) and
        hasattr(model, 'config') and
        isinstance(model.config, PretrainedConfig)
    ):
        propagate_config(model, 'config', model.config)
        convert(model, inplace=True,
                custom_module_class_mapping=HF_TRANSFORMER_MODULE_MAPPINGS)
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

    # TODO: better way to handle this?
    if args.activation is None:
        args.quantize_forward = None
    if args.error is None:
        args.quantize_backprop = None

    propagate_config(model, 'qconfig', qconfig)
    convert(model, DEFAULT_QAT_MODULE_MAPPINGS, inplace=True)
    prepare(model, True, args.quantize_forward, args.quantize_backprop, args.op_fusion)

    if run_fn is not None:
        run_fn(model)

    # TODO: use a config dict instead of argparse Namespace
    if hasattr(args, 'bf16') and args.bf16:
        model.bfloat16()

    # ASPLOS experiments perform quantization after converting model dtype to bfloat16
    # for name, param in model.named_parameters():
    #     if not 'bias' in name:
    #         param.data = qconfig.weight(device=param.device)(param.data)

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

def _register_module_hook(module, name, hook_name):
    module.add_module(hook_name, nn.ModuleDict())
    def _forward_pre_hook(self, inputs):
        new_inputs = []
        module_dict = getattr(self, hook_name)
        for i, input in enumerate(inputs):
            if not torch.is_tensor(input):
                new_inputs.append(input)
                continue

            if (obs_or_fq_name := str(i)) not in module_dict:
                obs_or_fq = getattr(self.qconfig, hook_name.split('_')[0])
                module_dict.update({
                    obs_or_fq_name: obs_or_fq(device=input.device, name=name)
                })
            new_inputs.append(module_dict[obs_or_fq_name](input))
        return tuple(new_inputs)

    def _backward_hook(self, grad_inputs, grad_outputs):
        return _forward_pre_hook(self, grad_inputs)

    if hook_name == 'activation_pre_process':
        module.register_forward_pre_hook(_forward_pre_hook)
    elif hook_name == 'error_pre_process':
        # backward pre hook has the same signature as forward pre hook
        module.register_full_backward_pre_hook(_forward_pre_hook)
    elif hook_name == 'error_post_process':
        module.register_full_backward_hook(_backward_hook)

def _add_observer_(
        module, fwd_pre_hook_module_list, bwd_pre_hook_module_list,
        bwd_residual, op_fusion, prefix):
    def _insert_obs_or_fq(m, name):
        if (op_fusion is not None and any(layer in name for layer in op_fusion)
            or not (hasattr(m, 'qconfig') and m.qconfig is not None)):
            return
        if isinstance(m, fwd_pre_hook_module_list):
            _register_module_hook(m, name, 'activation_pre_process')
        if isinstance(m, bwd_pre_hook_module_list):
            _register_module_hook(m, name, 'error_pre_process')
        is_residual = (
            isinstance(m, tuple(QCONFIG_PROPAGATE_MODULE_CLASS_LIST["residual"]))
            or (isinstance(m, tuple(QCONFIG_PROPAGATE_MODULE_CLASS_LIST["gemm"]))
                and any(layer in name for layer in RESIDUAL_LAYERS))
        )
        if bwd_residual and is_residual:
            _register_module_hook(m, name, 'error_post_process')

    named_modules = dict(module.named_children())
    for name, child in named_modules.items():
        module_prefix = prefix + '.' + name if prefix else name
        if isinstance(child, nni._FusedModule):
            _insert_obs_or_fq(child, module_prefix)
        else:
            _add_observer_(
                child, fwd_pre_hook_module_list, bwd_pre_hook_module_list,
                bwd_residual, op_fusion, module_prefix)
    _insert_obs_or_fq(module, prefix)

def prepare(
        model, inplace=False, ops_fwd=None, ops_bwd=None, op_fusion=None):
    if not inplace:
        model = copy.deepcopy(model)

    _add_observer_(
        model, _parse_ops(ops_fwd), _parse_ops(ops_bwd),
        ops_bwd and "residual" in ops_bwd, op_fusion, '')
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
        for pre_hook_fn in mod._backward_pre_hooks.values():
            new_mod.register_full_backward_pre_hook(pre_hook_fn)
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
            new_mod = Softmax(
                posit_exp, posit_exp_shifted, posit_reciprocal, dim=-1, dtype=dtype, device=device
            )
            setattr(module, name, new_mod)
        else:
            replace_softmax(mod, posit_exp, posit_exp_shifted, posit_reciprocal, dtype, device)

def get_quantized_model(model, qconfig, op_fusion=None, device=None):
    logger.info(f"Fusing operations: {op_fusion}")

    if device is None:
        devices = _get_unique_devices_(model)
        assert len(devices) <= 1, (
            f"_add_observer_ only works with cpu or single-device CUDA modules, but got devices {devices}"
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