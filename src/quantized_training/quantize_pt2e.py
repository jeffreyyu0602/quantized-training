import re
from typing import Dict

import torch
from torch import nn
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix, assert_and_get_unique_device
from torch.fx import GraphModule, Graph, Node

from .qconfig_mapping import (
    _OBJECT_TYPE_DICT_KEY,
    _OBJECT_TYPE_ORDER_DICT_KEY,
    _MODULE_NAME_DICT_KEY,
    _MODULE_NAME_REGEX_DICT_KEY,
    _MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY,
)


def quantize_pt2e(
        model, args, qconfig_mapping, example_args, example_kwargs=None,
        dynamic_shapes=None, run_fn=None):
    object_type_mappings = qconfig_mapping.object_type_qconfigs
    for name, module in model.named_modules():
        if (
            (qconfig := object_type_mappings.get(type(module))) is not None
            and type(qconfig.weight) != nn.Identity
        ):
            weight_fake_quant = qconfig.weight(device=module.weight.device)
            weight_fake_quant(module.weight)
            module.weight.data = weight_fake_quant(module.weight.data)

    # torch.export does not support training due to inplace operations
    # from torch.export import export
    # exported_program: torch.export.ExportedProgram = export(
    #     model,
    #     args=example_args,
    #     kwargs=example_kwargs,
    #     dynamic_shapes=dynamic_shapes,
    # )
    # model = prepare_pt2e(exported_program.module(), qconfig_mapping)

    from torch._export import capture_pre_autograd_graph
    model = capture_pre_autograd_graph(
        model,
        args=example_args,
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )
    model = prepare_pt2e(model, qconfig_mapping)

    if hasattr(args, 'bf16') and args.bf16:
        model.bfloat16()

    if args.activation and args.activation.qscheme and run_fn is not None:
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


def _match_fields(fields, patterns):
    if len(fields) != len(patterns):
        return False

    for pattern, value in zip(patterns, fields):
        if pattern == value:
            continue
        if isinstance(pattern, str):
            try:
                if re.fullmatch(pattern, value):
                    continue
            except re.error:
                pass
        return False
    return True


def _get_qconfig(fields, mappings, qconfig=None):
    if qconfig is not None:
        return qconfig
    for mapping in mappings:
        # TODO: enforce pattern type
        if _match_fields(fields, mapping[:-1]):  # the last element is the qconfig
            return mapping[-1]
    return qconfig


def _get_qconfig_for_node(node, order, qconfig_mapping):
    if not isinstance(node, Node):
        return None
    qconfig_dict = qconfig_mapping.to_dict()
    qconfig = _get_qconfig((node.name, node.target, order), qconfig_dict[_MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY])
    qconfig = _get_qconfig((node.name,), qconfig_dict[_MODULE_NAME_DICT_KEY], qconfig)
    qconfig = _get_qconfig((node.name,), qconfig_dict[_MODULE_NAME_REGEX_DICT_KEY], qconfig)
    qconfig = _get_qconfig((node.target, order), qconfig_dict[_OBJECT_TYPE_ORDER_DICT_KEY], qconfig)
    qconfig = _get_qconfig((node.target,), qconfig_dict[_OBJECT_TYPE_DICT_KEY], qconfig)
    return qconfig


def prepare_pt2e(model, qconfig_mapping):
    observed_ops = []
    unobserved_ops = []
    # Go through all the nodes in the Graph
    nodes_before_observation = list(model.graph.nodes)
    for node in nodes_before_observation:
        named_modules = dict(model.named_modules(remove_duplicate=False))
        new_args = []
        args_updated = False
        for i, arg in enumerate(node.args):
            qconfig = _get_qconfig_for_node(node, i, qconfig_mapping)
            if qconfig is None:
                new_args.append(arg)
                continue
            args_updated = True
            observed_ops.append(str(node.target))
            # PyTorch fake quant classes do not accept name as an argument
            try:
                input_obs_or_fq = qconfig.activation(name=f"{node.name}.{i}")
            except TypeError:
                input_obs_or_fq = qconfig.activation()
            new_arg = _insert_obs_or_fq(arg, input_obs_or_fq, model, named_modules, model.graph)
            new_args.append(new_arg)

        if args_updated:
            node.args = tuple(new_args)
        elif node.op == 'call_function':
            unobserved_ops.append(str(node.target))

    print("=" * 40)
    print("Observed ops: ")
    print('\n'.join(list(set(observed_ops))))
    print("=" * 40)
    print("Unobserved ops: ")
    print('\n'.join(list(set(unobserved_ops))))
    return GraphModule(model, model.graph)