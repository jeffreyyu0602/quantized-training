import re
from typing import Dict

import torch
from torch import nn
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix, assert_and_get_unique_device
from torch.export import export
from torch.fx import GraphModule, Graph, Node

from .fake_quantize import FusedAmaxObsFakeQuantize
from .qconfig import QConfig
from .quantization_mappings import QUANTIZATION_OPERATORS


def _quantize_weight(float_wt, quant_min=-128, quant_max=127):
    min_val, max_val = torch.aminmax(float_wt)
    max_val = torch.max(-min_val, max_val)
    scale = max_val / (float(quant_max - quant_min) / 2)
    return torch.clamp(torch.round(float_wt / scale), quant_min, quant_max) * scale


def quantize_pt2e(model, args, example_args, example_kwargs=None, dynamic_shapes=None, run_fn=None):
    if args.dtype in ["qint8", "quint8"]:
        from torch.ao.quantization import (
            MovingAverageMinMaxObserver,
            MovingAveragePerChannelMinMaxObserver,
            FusedMovingAvgObsFakeQuantize,
            default_weight_observer,
        )
        from torch.ao.quantization.fake_quantize import _is_per_channel
        qscheme = (
            getattr(torch, args.scaling_fwd[0])
            if args.scaling_fwd[0] is not None
            else torch.per_tensor_affine
        )
        observer = (
            MovingAveragePerChannelMinMaxObserver
            if _is_per_channel(qscheme) else MovingAverageMinMaxObserver
        )
        default_act_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(
            observer=observer,
            ch_axis=1,
            quant_min=-128,
            quant_max=127,
            dtype=getattr(torch, args.dtype),
            qscheme=qscheme,
        )
        qconfig = torch.ao.quantization.QConfig(
            activation=default_act_fake_quant if args.quantize_fwd else nn.Identity,
            weight=default_weight_observer if args.quantize_weight else nn.Identity,
        )

        for name, param in model.named_parameters():
            if not 'bias' in name:
                param.data = _quantize_weight(param.data, -128, 127)
    else:
        if "," in args.dtype:
            dtype_fwd, dtype_bwd = args.dtype.split(",")
        elif re.search(r'^FP8(\.MIXED)?$', args.dtype, re.IGNORECASE):
            dtype_fwd, dtype_bwd = ("E4M3", "E5M2")
        else:
            dtype_fwd, dtype_bwd = (args.dtype, args.dtype)

        default_fake_quant = FusedAmaxObsFakeQuantize.with_args(
            dtype=dtype_fwd,
            qscheme=args.scaling_fwd[0],
            quant_max=args.scaling_fwd[1],
            amax_history_len=args.scaling_fwd[2],
            record_histogram=args.record_histogram
        )

        error_fake_quant = FusedAmaxObsFakeQuantize.with_args(
            dtype=dtype_bwd,
            qscheme=args.scaling_bwd[0],
            quant_max=args.scaling_bwd[1],
            amax_history_len=args.scaling_bwd[2],
            record_histogram=args.record_histogram
        )

        qconfig = QConfig(
            activation=default_fake_quant if args.quantize_fwd else nn.Identity,
            weight=default_fake_quant if args.quantize_weight else nn.Identity,
            error=error_fake_quant if args.quantize_bwd else nn.Identity,
        )

        for name, param in model.named_parameters():
            if not 'bias' in name:
                param.data = qconfig.weight(device=param.device)(param.data)

    ops = args.quantize_fwd.split(',') if args.quantize_fwd is not None else []
    node_list = tuple(node for op in ops for node in QUANTIZATION_OPERATORS[op.lower()])

    # torch.export does not support training due to inplace operations
    # exported_program: torch.export.ExportedProgram = export(
    #     model,
    #     args=example_args,
    #     kwargs=example_kwargs,
    #     dynamic_shapes=dynamic_shapes,
    # )
    # model = prepare_pt2e(exported_program.module(), node_list, qconfig=qconfig)

    # from torch.export._trace import _export
    # ep = _export(model, example_args, example_kwargs, dynamic_shapes, pre_dispatch=True)
    # model = prepare_pt2e(ep.module(), node_list, qconfig=qconfig)

    from torch._export import capture_pre_autograd_graph
    model = capture_pre_autograd_graph(
        model,
        args=example_args,
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )
    model = prepare_pt2e(model, node_list, qconfig=qconfig)

    if hasattr(args, 'bf16') and args.bf16:
        model.bfloat16()

    if args.scaling_fwd[0] and run_fn is not None:
        run_fn(model)

    return model


def _insert_obs_or_fq(
    node: Node,
    obs_or_fq: ObserverOrFakeQuantize,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
) -> Node:
    """
    Attaches `obs_or_fq` to `model`, and creates a node which calls
    `obs_or_fq` on the output of `node`.

    obs_or_fq: an instance of Observer or FakeQuantize module
    """
    model_device = assert_and_get_unique_device(model)
    if model_device:
        obs_or_fq.to(model_device)
    # add obs_or_fq module as attribute
    get_new_obs_or_fq_name = get_new_attr_name_with_prefix('activation_pre_process_')
    obs_or_fq_name = get_new_obs_or_fq_name(model)
    setattr(model, obs_or_fq_name, obs_or_fq)
    named_modules[obs_or_fq_name] = obs_or_fq
    with graph.inserting_after(node):
        new_obs = graph.create_node(
            'call_module', obs_or_fq_name, (node,), {})
    return new_obs


def prepare_pt2e(model, fwd_obs_or_fq_node_list, qconfig=None):
    observed_ops = []
    unobserved_ops = []
    # Go through all the nodes in the Graph
    nodes_before_observation = list(model.graph.nodes)
    for node in nodes_before_observation:
        matching_pattern = None
        for pattern in fwd_obs_or_fq_node_list:
            if node.target == (pattern[0] if isinstance(pattern, tuple) else pattern):
                matching_pattern = pattern
                break

        if matching_pattern is not None:
            observed_ops.append(str(node.target))
            named_modules = dict(model.named_modules(remove_duplicate=False))
            new_args = []
            for i, arg in enumerate(node.args):
                if (
                    not isinstance(arg, Node) or
                    isinstance(matching_pattern, tuple)
                    and i not in matching_pattern[1]
                ):
                    new_args.append(arg)
                    continue
                # TODO: fake quant class does not accept name argument
                input_obs_or_fq = qconfig.activation()
                new_arg = _insert_obs_or_fq(arg, input_obs_or_fq, model, named_modules, model.graph)
                new_args.append(new_arg)
            node.args = tuple(new_args)
        elif node.op == 'call_function':
            unobserved_ops.append(str(node.target))

        # TODO: handle backward residual
        if len(list(node.users)) > 1:
            pass

    print("=" * 80)
    print("Observed ops: ")
    print('\n'.join(list(set(observed_ops))))
    print("=" * 80)
    print("Unobserved ops: ")
    print('\n'.join(list(set(unobserved_ops))))
    return GraphModule(model, model.graph)