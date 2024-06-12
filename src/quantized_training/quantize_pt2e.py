import copy
from dataclasses import asdict, replace
from typing import Optional, Any, Callable

import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.fx.utils import assert_and_get_unique_device

from torch.ao.quantization.fx.prepare import (
    _insert_obs_or_fq,
    _save_state,
    _is_activation_post_process_node,
    _create_obs_or_fq_from_qspec,
)
from torch.fx import (
    GraphModule,
    Graph,
    Node,
)
from torch.fx.node import Argument

from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from typing import Dict, Tuple, Union, Any, Optional
from torch.ao.quantization.quantizer import (
    EdgeOrNode,
    SharedQuantizationSpec,
    QuantizationSpecBase,
)
from torch.ao.quantization import ObserverOrFakeQuantize

import quantized_training as qt
from quantized_training.export_utils import _allow_exported_model_train_eval
from quantized_training.fake_quantize import FusedAmaxObsFakeQuantize, _get_fake_quant_fn
from quantized_training.quantizer.quantizer import QuantizationSpec
from quantized_training.quantizer.xnnpack_quantizer import XNNPACKQuantizer
from quantized_training.quantizer.xnnpack_quantizer_utils import QuantizationConfig

from .decomposed import quantized_decomposed_lib


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
    kwargs_dict = asdict(quantization_spec)
    kwargs = copy.deepcopy(kwargs_dict)
    kwargs.pop("observer_or_fake_quant_ctr")
    return observer_or_fake_quant_ctr.with_args(**kwargs)()


def _get_obs_or_fq_map(
    edge_or_node_to_group_id: Dict[EdgeOrNode, int],
    edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase],
    is_qat: bool
) -> Dict[EdgeOrNode, ObserverOrFakeQuantize]:
    """Generates the EdgeOrNode to observer/fake_quant instances
    Makes sure that for EdgeOrNode that has the same group_id should have the same observer or fake quant
    instances
    """
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize] = {}
    group_id_to_obs_or_fq: Dict[int, ObserverOrFakeQuantize] = {}
    for edge_or_node, qspec in edge_or_node_to_qspec.items():
        group_id = edge_or_node_to_group_id[edge_or_node]
        if group_id not in group_id_to_obs_or_fq:
            # TODO: maybe edge_or_node_to_qspec should be edge_or_node_to_root_qspec, this will simplify
            # the implementation for _create_obs_or_fq_from_qspec
            group_id_to_obs_or_fq[group_id] = _create_obs_or_fq_from_qspec(qspec, obs_or_fq_map, is_qat)
        obs_or_fq_map[edge_or_node] = group_id_to_obs_or_fq[group_id]
    return obs_or_fq_map


def get_microscaling_quantizer(
    activation: Optional[QuantizationSpec],
    weight: Optional[QuantizationSpec],
):
    # Microscaling performs quantization along the reduction dimension
    act_qspec = replace(activation, ch_axis=1)
    weight_qspec = replace(weight, ch_axis=1)
    qconfig_conv2d = QuantizationConfig(act_qspec, None, weight_qspec, None)

    act_qspec = replace(activation, ch_axis=-1)
    weight_qspec = replace(weight, ch_axis=-1)
    qconfig_linear = QuantizationConfig(act_qspec, None, weight_qspec, None)

    act0_qspec = replace(activation, ch_axis=-1)
    act1_qspec = replace(activation, ch_axis=-2)
    qconfig_matmul = QuantizationConfig(act0_qspec, None, act1_qspec, None)

    quantizer = XNNPACKQuantizer() \
        .set_module_type(torch.nn.Conv2d, qconfig_conv2d) \
        .set_module_type(torch.nn.Linear, qconfig_linear) \
        .set_operator_type(torch.ops.aten.matmul.default, qconfig_matmul)
    return quantizer


def get_per_channel_act_quantizer(
    activation: Optional[QuantizationSpec],
    weight: Optional[QuantizationSpec],
):
    # Convolution layer only support per-tensor activation quantization
    act_qspec = replace(activation, qscheme=qt.per_tensor_symmetric)
    qconfig_conv2d = QuantizationConfig(act_qspec, None, weight, None)

    # Perform quantization along the outer dimension
    act_qspec = replace(activation, ch_axis=-2)
    qconfig_linear = QuantizationConfig(act_qspec, None, weight, None)

    act0_qspec = replace(activation, ch_axis=-2)
    act1_qspec = replace(activation, ch_axis=-1)
    qconfig_matmul = QuantizationConfig(act0_qspec, None, act1_qspec, None)

    quantizer = XNNPACKQuantizer() \
        .set_module_type(torch.nn.Conv2d, qconfig_conv2d) \
        .set_module_type(torch.nn.Linear, qconfig_linear) \
        .set_operator_type(torch.ops.aten.matmul.default, qconfig_matmul)
    return quantizer


def get_quantizer(
    activation: Optional[QuantizationSpec],
    weight: Optional[QuantizationSpec],
    record_histogram: bool = False,
    force_scale_power_of_two: bool = False
) -> XNNPACKQuantizer:
    """
    Create a quantizer for the given activation and weight quantization specifications.

    Parameters:
    - activation: The quantization spec for activations.
    - weight: The quantization spec for weights.
    - record_histogram: Whether to record histogram of input.
    - force_scale_power_of_two: Whether to force the scaling factor to be a power of two.

    Returns:
    - A configured XNNPACKQuantizer.
    """

    observer_or_fake_quant_ctr = FusedAmaxObsFakeQuantize.with_args(
        record_histogram=record_histogram,
        force_scale_power_of_two=force_scale_power_of_two,
    )

    qschemes = []
    if activation is not None:
        qschemes.append(activation.qscheme)
        activation.observer_or_fake_quant_ctr = observer_or_fake_quant_ctr
    if weight is not None:
        qschemes.append(weight.qscheme)
        weight.observer_or_fake_quant_ctr = observer_or_fake_quant_ctr

    if qt.per_vector_symmetric in qschemes or qt.microscaling in qschemes:
        assert len(set(qschemes)) == 1, (
            f"Quantization scheme {qschemes[0]} does not work with {qschemes[1]}"
        )
        return get_microscaling_quantizer(activation, weight)

    if weight is not None and weight.qscheme == qt.per_channel_symmetric:
        assert weight.ch_axis == 0, (
            f"Per-channel weight quantization only supports quantizing output "
            "channel dimension (dim=0)."
        )

    if activation is not None and activation.qscheme == qt.per_channel_symmetric:
        return get_per_channel_act_quantizer(activation, weight)

    qconfig = QuantizationConfig(activation, None, weight, None)
    qconfig_matmul = QuantizationConfig(activation, None, activation, None)
    return XNNPACKQuantizer().set_global(qconfig) \
        .set_operator_type(torch.ops.aten.matmul.default, qconfig_matmul)


def prepare_pt2e(model, quantizer, args, kwargs=None, dynamic_shapes=None):
    from torch.ao.quantization.pt2e import prepare
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e

    # HACK monkey patching to replace the default implementation of _create_obs_or_fq_from_qspec
    prepare._get_obs_or_fq_map = _get_obs_or_fq_map

    # Make sure captured model is on the same device as the original model
    model_device = assert_and_get_unique_device(model)
    model = capture_pre_autograd_graph(
        model,
        args=args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
    ).to(model_device)

    model = prepare_pt2e(model, quantizer)
    _allow_exported_model_train_eval(model)
    return model


def _get_module(node: Node, named_modules: Dict[str, torch.nn.Module]) -> Optional[torch.nn.Module]:
    """
    If `node` refers to a call_module node, return the module, else None.
    """
    if node.op == "call_module" and str(node.target) in named_modules:
        return named_modules[str(node.target)]
    else:
        return None


# Returns a function that can get a new attribute name for module with given
# prefix, for example,
# >> get_new_observer_name = get_new_attr_name_with_prefix('_observer')
# >> new_name = get_new_observer_name(module)
# new_name will be an unused attribute name on module, e.g. `_observer_1`
def get_new_attr_name_with_prefix(prefix: str) -> Callable:
    prefix = prefix.replace(".", "_")

    def get_new_attr_name(module: torch.nn.Module):
        def get_attr_name(i: int):
            return prefix + str(i)
        i = 0
        attr_name = get_attr_name(i)
        while hasattr(module, attr_name):
            i += 1
            attr_name = get_attr_name(i)
        return attr_name
    return get_new_attr_name


def create_getattr_from_value(module: torch.nn.Module, graph: Graph, prefix: str, value: Any) -> Node:
    """
    Given a value of any type, creates a getattr node corresponding to the value and
    registers the value as a buffer to the module.
    """
    get_new_attr_name = get_new_attr_name_with_prefix(prefix)
    attr_name = get_new_attr_name(module)
    device = assert_and_get_unique_device(module)
    new_value = value.clone().detach() if isinstance(value, torch.Tensor) \
        else torch.tensor(value, device=device)
    module.register_buffer(attr_name, new_value)
    # Create get_attr with value
    attr_node = graph.create_node("get_attr", attr_name)
    return attr_node


def _fuse_dequantize_quantize(model: GraphModule):
    from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions

    dequantize_quantize_pattern = [
        torch.ops.quantized_ops.dequantize_symmetric,
        torch.ops.quantized_ops.quantize_symmetric,
    ]
    partitions = find_sequential_partitions(
        model, dequantize_quantize_pattern, filter_fn=None
    )
    for partition in partitions:
        dq_partition, q_partition = partition
        print(dq_partition.nodes)
        print(q_partition.nodes)
        print()
    return model


def convert_pt2e(model: GraphModule):
    modules = dict(model.named_modules(remove_duplicate=False))
    graph = model.graph

    for node in list(model.graph.nodes):
        if node.op == "call_module":
            mod = _get_module(node, modules)
            assert mod is not None
            if isinstance(mod, torch.ao.quantization.FakeQuantizeBase):
                # reshape weight scale to match the shape of the output tensor.
                # This is necessary for per-channel weight quantization.
                # TODO: check node is a weight node and handle per-token activation quantization
                scale = mod.scale
                if scale.ndim == 4:
                    scale = scale.view(1, -1, 1, 1)
                elif scale.ndim == 2:
                    scale = scale.view(1, -1)

                # replace fake quant module with a quantize node
                with graph.inserting_before(node):
                    qparam_node = create_getattr_from_value(
                        model, graph, next(iter(node.users)).name + "_scale_", mod.scale)
                    qvalues_node = create_getattr_from_value(
                        model, graph, "qvalues_", mod.qvalues)
                    quantize_op_inputs = [node.args[0], qparam_node, mod.dtype, qvalues_node]
                    quantized_node = graph.call_function(
                        torch.ops.quantized_ops.quantize_symmetric,
                        tuple(quantize_op_inputs),
                        {}
                    )
                node.replace_all_uses_with(quantized_node)
                graph.erase_node(node)

                # insert a dequantize node after each user
                orig_quantized_node_users = list(quantized_node.users.keys())
                for user_node in orig_quantized_node_users:
                    maybe_dq_node = next(iter(user_node.users))
                    if (
                        maybe_dq_node.op != "call_function"
                        or maybe_dq_node.target != torch.ops.quantized_ops.dequantize_symmetric
                    ):
                        with graph.inserting_before(maybe_dq_node):
                            output_qparam_node = create_getattr_from_value(
                                model, graph, user_node.name + "_dq_scale_", scale)
                            dq_inputs = [user_node, output_qparam_node, mod.dtype, qvalues_node]
                            dequantized_node = graph.call_function(
                                torch.ops.quantized_ops.dequantize_symmetric,
                                tuple(dq_inputs),
                                {}
                            )

                        # We need to save orig users before updating uses because
                        # the list of users will change as we update uses
                        orig_users = list(user_node.users.keys())
                        for user in orig_users:
                            if id(user) == id(dequantized_node):
                                continue
                            user.replace_input_with(user_node, dequantized_node)
                    else:
                        # update scale in the dequantize node
                        output_qparam_node = maybe_dq_node.args[1]
                        scale = scale * model.get_buffer(output_qparam_node.target)
                        model.register_buffer(output_qparam_node.target, scale)

                    if user_node.op != 'call_function' or user_node.target not in [
                        torch.ops.aten.conv2d.default,
                        torch.ops.aten.linear.default,
                    ]:
                        continue

                    # Scale bias using the product of input activation scale and weight scale.
                    # It shares the same qparam and qvalues node with the dequantize node.
                    maybe_bias_node = user_node.args[2]
                    if (
                        maybe_bias_node.op != 'call_function'
                        or maybe_bias_node.target != torch.ops.quantized_ops.quantize_symmetric
                    ):
                        # use int16/int32 bias for int8 quantization and floating-point bias otherwise
                        # int16 bias has bad accuracy for per-channel weight quantization
                        device = assert_and_get_unique_device(mod)
                        qvalues = torch.arange(2 ** 16, dtype=torch.int16, device=device).view(torch.bfloat16)
                        if mod.dtype == "int8":
                            fq_fn = _get_fake_quant_fn("int32")
                            qvalues = fq_fn(qvalues)
                        with graph.inserting_before(user_node):
                            bias_qparam_node = create_getattr_from_value(
                                model, graph, user_node.name + "_bias_scale_", scale.flatten())
                            qvalues_node = create_getattr_from_value(model, graph, "qvalues_", qvalues)
                            qb_inputs = [maybe_bias_node, bias_qparam_node, mod.dtype, qvalues_node]
                            quantized_bias_node = graph.call_function(
                                torch.ops.quantized_ops.quantize_symmetric,
                                tuple(qb_inputs),
                                {},
                            )

                        orig_users = list(maybe_bias_node.users.keys())
                        for user in orig_users:
                            if id(user) == id(quantized_bias_node):
                                continue
                            user.replace_input_with(maybe_bias_node, quantized_bias_node)
                    else:
                        bias_qparam_node = maybe_bias_node.args[1]
                        model.register_buffer(bias_qparam_node.target, scale.flatten())

    model.graph.print_tabular()

    _fuse_dequantize_quantize(model)

    model.graph.lint()
    model.recompile()
    return model
