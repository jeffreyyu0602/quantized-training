import re

import torch
from torch import nn
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix, assert_and_get_unique_device
from torch.export import export
from torch.fx import GraphModule, Node
from functorch.compile import aot_function

from .fake_quantize import FusedAmaxObsFakeQuantize
from .qconfig import QConfig
from .quantization_mappings import QUANTIZATION_OPERATORS


def quantize_fx(model, args, example_args, example_kwargs=None):
    if "," in args.dtype:
        dtype_fwd, dtype_bwd = args.dtype.split(",")
    elif re.search(r'^FP8(\.MIXED)?$', args.dtype, re.IGNORECASE):
        dtype_fwd, dtype_bwd = ("E4M3", "E5M2")
    else:
        dtype_fwd, dtype_bwd = (args.dtype, args.dtype)

    default_fake_quant = FusedAmaxObsFakeQuantize.with_args(
        dtype=dtype_fwd,
        is_per_tensor=args.scaling_fwd,
        quant_max=args.max_fwd,
        amax_history_len=args.amax_history_len,
        observer_enabled=args.record_histogram
    )

    error_fake_quant = FusedAmaxObsFakeQuantize.with_args(
        dtype=dtype_bwd,
        is_per_tensor=args.scaling_bwd,
        quant_max=args.max_bwd,
        amax_history_len=args.amax_history_len,
        observer_enabled=args.record_histogram
    )

    qconfig = QConfig(
        activation=default_fake_quant if args.quantize_fwd else nn.Identity,
        weight=default_fake_quant if args.quantize_weights else nn.Identity,
        error=error_fake_quant if args.quantize_bwd else nn.Identity,
    )

    ops = args.quantize_fwd.split(',') if args.quantize_fwd is not None else []
    patterns = tuple(mod for op in ops for mod in QUANTIZATION_OPERATORS[op.lower()])
    model = prepare(model, patterns, example_args, example_kwargs, qconfig=qconfig)

    if hasattr(args, 'bf16') and args.bf16:
        model.bfloat16()

    for name, param in model.named_parameters():
        if not 'bias' in name:
            param.data = qconfig.weight(device=param.device)(param.data)

    return model


def prepare(module, patterns, example_args, example_kwargs=None, dynamic_shapes=None, qconfig=None):
    exported_program: torch.export.ExportedProgram = export(
        module,
        args=example_args,
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )
    model = exported_program.module()

    observed_ops = []
    unobserved_ops = []
    named_modules = {}
    # Go through all the nodes in the Graph
    for node in model.graph.nodes:
        matching_pattern = None
        for pattern in patterns:
            if node.target == (pattern[0] if isinstance(pattern, tuple) else pattern):
                matching_pattern = pattern
                break

        if matching_pattern is not None:
            observed_ops.append(str(node.target))
            new_args = []
            for i, arg in enumerate(node.args):
                if (
                    isinstance(matching_pattern, tuple)
                    and i not in matching_pattern[1]
                    or not isinstance(arg, Node)
                ):
                    new_args.append(arg)
                    continue
                obs_or_fq = qconfig.activation(name=node.name)
                model_device = assert_and_get_unique_device(model)
                if model_device:
                    obs_or_fq.to(model_device)
                # add obs_or_fq module as attribute
                prefix = 'activation_pre_process_'
                get_new_obs_or_fq_name = get_new_attr_name_with_prefix(prefix)
                obs_or_fq_name = get_new_obs_or_fq_name(model)
                setattr(model, obs_or_fq_name, obs_or_fq)
                named_modules[obs_or_fq_name] = obs_or_fq
                with model.graph.inserting_after(arg):
                    new_node = model.graph.call_module(obs_or_fq_name, (arg,))
                new_args.append(new_node)
            node.args = tuple(new_args)
        elif node.op == 'call_function':
            unobserved_ops.append(str(node.target))

    print("=" * 80)
    print("Observed ops: ")
    print('\n'.join(list(set(observed_ops))))
    print("=" * 80)
    print("Unobserved ops: ")
    print('\n'.join(list(set(unobserved_ops))))
    return GraphModule(model, model.graph)
