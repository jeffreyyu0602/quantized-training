from typing import Dict

import torch
from torch import nn
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix, assert_and_get_unique_device
from torch.fx import GraphModule, Graph, Node

from .qconfig_mapping import (
    _OBJECT_TYPE_DICT_KEY,
    _OBJECT_TYPE_ORDER_DICT_KEY,
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


def _get_matched_qconfig(node, field, mappings, match=None):
    if match is not None:
        return match
    for mapping in mappings:
        if getattr(node, field) == mapping[0]:
            return mapping
    return match


def prepare_pt2e(model, qconfig_mapping):
    observed_ops = []
    unobserved_ops = []
    qconfig_dict = qconfig_mapping.to_dict()
    # Go through all the nodes in the Graph
    nodes_before_observation = list(model.graph.nodes)
    for node in nodes_before_observation:
        # TODO: check all the patterns
        # Match primary field
        match = _get_matched_qconfig(node, 'name', qconfig_dict[_MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY])
        match = _get_matched_qconfig(node, 'target', qconfig_dict[_OBJECT_TYPE_ORDER_DICT_KEY], match)
        match = _get_matched_qconfig(node, 'target', qconfig_dict[_OBJECT_TYPE_DICT_KEY], match)
        if match is not None:
            observed_ops.append(str(node.target))
            named_modules = dict(model.named_modules(remove_duplicate=False))
            new_args = []
            for i, arg in enumerate(node.args):
                if (
                    not isinstance(arg, Node)
                    or len(match) > 2 and match[-2] != i # argument order doesn't match
                ):
                    new_args.append(arg)
                    continue
                qconfig = match[-1]
                # PyTorch fake quant class does not accept name as an argument
                try:
                    input_obs_or_fq = qconfig.activation(name=f"{node.name}.{i}")
                except TypeError:
                    input_obs_or_fq = qconfig.activation()
                new_arg = _insert_obs_or_fq(arg, input_obs_or_fq, model, named_modules, model.graph)
                new_args.append(new_arg)
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