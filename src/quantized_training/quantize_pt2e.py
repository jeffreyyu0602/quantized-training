import re
from typing import Dict

import torch
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix, assert_and_get_unique_device
from torch.fx import GraphModule, Graph, Node
from torch._export import capture_pre_autograd_graph

from .fake_quantize import FusedAmaxObsFakeQuantize
from .quantizer import QuantizationSpec
from .quantizer.xnnpack_quantizer import QuantizationConfig, XNNPACKQuantizer

from .qconfig_mapping import (
    _OBJECT_TYPE_DICT_KEY,
    _OBJECT_TYPE_ORDER_DICT_KEY,
    _MODULE_NAME_DICT_KEY,
    _MODULE_NAME_REGEX_DICT_KEY,
    _MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY,
)


def _create_obs_or_fq_from_qspec(quantization_spec, obs_or_fq_map, is_qat):
    """ Create observer or fake quantize objects based on quantization spec

    Args:
       quantization_spec: used to store parameters to create the observer or fake quantizer
       obs_or_fq_map: this is a map from edge/output to the corresponding observer/fake_quant
       instance, it may be reused for different edge/output depending on configuration
    """
    if quantization_spec is None:
        return None

    assert isinstance(quantization_spec, QuantizationSpec)
    observer_or_fake_quant_ctr = quantization_spec.observer_or_fake_quant_ctr
    kwargs = quantization_spec.to_dict()
    kwargs.pop("observer_or_fake_quant_ctr")
    return FusedAmaxObsFakeQuantize.with_args(**kwargs)()


def prepare(
        model, args, example_args, example_kwargs=None, dynamic_shapes=None):
    # TODO: monkey patching to replace the default implementation of _create_obs_or_fq_from_qspec
    import torch.ao.quantization.fx.prepare
    import sys
    sys.modules["torch.ao.quantization.fx.prepare"]._create_obs_or_fq_from_qspec = \
        _create_obs_or_fq_from_qspec
    
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e

    model = capture_pre_autograd_graph(
        model,
        args=example_args,
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )

    qconfig = QuantizationConfig(args.activation, args.error, args.weight, None)
    quantizer = XNNPACKQuantizer().set_global(qconfig)

    prepare_pt2e(model, quantizer)
    model.recompile()
    model.graph.print_tabular()
    return model


# TODO: this function will be replaced by prepare in the future
def quantize_pt2e(
        model, args, qconfig_mapping, example_args, example_kwargs=None,
        dynamic_shapes=None):
    for name, module in model.named_modules():
        qconfig = qconfig_mapping.object_type_qconfigs.get(type(module))
        if qconfig is not None:
            obs_or_fq = qconfig.weight(device=module.weight.device, name=name)
            obs_or_fq(module.weight)
            module.weight.data = obs_or_fq(module.weight.data)

    model = capture_pre_autograd_graph(
        model,
        args=example_args,
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )
    model = prepare_pt2e(model, qconfig_mapping)
    model.graph.print_tabular()

    if hasattr(args, 'bf16') and args.bf16:
        model.bfloat16()

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
