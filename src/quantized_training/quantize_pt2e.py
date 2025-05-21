import copy
import itertools
import operator
import re
from collections import OrderedDict
from dataclasses import asdict, replace
from typing import Dict, Tuple, Any, Optional, Callable, List

import torch
from torch import Tensor
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.fx.utils import assert_and_get_unique_device
from torch.ao.quantization.quantizer import (
    EdgeOrNode,
    SharedQuantizationSpec,
    QuantizationSpecBase,
)
from torch.fx import (
    GraphModule,
    Graph,
    Node,
)
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

import quantized_training as qt
from quantized_training.fake_quantize import (
    _DerivedObserverOrFakeQuantize,
    FusedAmaxObsFakeQuantize,
    get_quantization_map,
)
from quantized_training.quantizer.quantizer import QuantizationSpec, DerivedQuantizationSpec
from quantized_training.quantizer.xnnpack_quantizer import XNNPACKQuantizer
from quantized_training.quantizer.xnnpack_quantizer_utils import QuantizationConfig

from .codegen.mapping_utils import (
    _is_gemm_op,
    _is_indexing_or_concatenation_op,
    _is_nop,
    _is_reshape_op,
)
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
    if isinstance(quantization_spec, DerivedQuantizationSpec):
        kwargs = {
            "dtype": quantization_spec.dtype,
            "derive_qparams_fn": quantization_spec.derive_qparams_fn,
        }
        edge_or_nodes = quantization_spec.derived_from
        obs_or_fqs = [obs_or_fq_map[k] for k in edge_or_nodes]
        kwargs["obs_or_fqs"] = obs_or_fqs
        return _DerivedObserverOrFakeQuantize.with_args(**kwargs)()

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


def _set_ch_axis(qspec: Optional[QuantizationSpec], ch_axis: int):
    if qspec is None:
        return None
    return replace(qspec, ch_axis=ch_axis)


def get_microscaling_quantizer(
    activation: Optional[QuantizationSpec],
    weight: Optional[QuantizationSpec],
):
    # Microscaling performs quantization along the reduction dimension
    act_qspec = _set_ch_axis(activation, 1)
    weight_qspec = _set_ch_axis(weight, 1)
    qconfig_conv2d = QuantizationConfig(act_qspec, None, weight_qspec, None)

    act_qspec = _set_ch_axis(activation, -1)
    weight_qspec = _set_ch_axis(weight, -1)
    qconfig_linear = QuantizationConfig(act_qspec, None, weight_qspec, None)

    act0_qspec = _set_ch_axis(activation, -1)
    act1_qspec = _set_ch_axis(activation, -2)
    qconfig_matmul = QuantizationConfig(act0_qspec, None, act1_qspec, None)

    return (XNNPACKQuantizer()
        .set_module_type(torch.nn.Conv2d, qconfig_conv2d)
        .set_module_type(torch.nn.Linear, qconfig_linear)
        .set_object_type(torch.ops.aten.matmul.default, qconfig_matmul)
    )


def get_per_channel_act_quantizer(
    input_activation: Optional[QuantizationSpec],
    output_activation: Optional[QuantizationSpec],
    weight: Optional[QuantizationSpec],
    bias: Optional[QuantizationSpec],
):
    # Convolution layer only support per-tensor activation quantization
    act_qspec = replace(input_activation, qscheme=qt.per_tensor_symmetric)
    qconfig_conv2d = QuantizationConfig(act_qspec, output_activation, weight, bias)

    # Perform quantization along the outer dimension
    act_qspec = replace(input_activation, ch_axis=-2)
    qconfig_linear = QuantizationConfig(act_qspec, output_activation, weight, bias)

    act0_qspec = replace(input_activation, ch_axis=-2)
    act1_qspec = replace(input_activation, ch_axis=-1)
    qconfig_matmul = QuantizationConfig(act0_qspec, output_activation, act1_qspec, None)

    return (XNNPACKQuantizer()
        .set_module_type(torch.nn.Conv2d, qconfig_conv2d)
        .set_module_type(torch.nn.Linear, qconfig_linear)
        .set_object_type(torch.ops.aten.matmul.default, qconfig_matmul)
    )


def derive_bias_qparams_fn(obs_or_fqs: List[ObserverOrFakeQuantize]) -> Tuple[Tensor, Tensor]:
    assert len(obs_or_fqs) == 2, \
        "Expecting two obs/fqs, one for activation and one for weight, got: {}".format(len(obs_or_fqs))
    act_obs_or_fq = obs_or_fqs[0]
    weight_obs_or_fq = obs_or_fqs[1]
    act_scale = act_obs_or_fq.calculate_qparams()
    weight_scale = weight_obs_or_fq.calculate_qparams()
    return act_scale * weight_scale.flatten()


def get_default_quantizer(
    input_activation: Optional[QuantizationSpec],
    output_activation: Optional[QuantizationSpec] = None,
    weight: Optional[QuantizationSpec] = None,
    bias: Optional[QuantizationSpec] = None,
    record_histogram: bool = False,
    force_scale_power_of_two: bool = False,
    **kwargs: Any,
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
    if input_activation is not None:
        input_activation = QuantizationSpec.from_str(input_activation)
        input_activation.observer_or_fake_quant_ctr = observer_or_fake_quant_ctr
        qschemes.append(input_activation.qscheme)

    if output_activation is not None:
        output_activation = QuantizationSpec.from_str(output_activation)
        output_activation.observer_or_fake_quant_ctr = observer_or_fake_quant_ctr

    if weight is not None:
        weight = QuantizationSpec.from_str(weight)
        weight.observer_or_fake_quant_ctr = observer_or_fake_quant_ctr
        qschemes.append(weight.qscheme)

    qschemes = [qs for qs in qschemes if qs is not None]
    if len(qschemes) > 0 and qt.microscaling not in qschemes:
        assert bias is not None, (
            "Bias quantization is required when quantizing activations and weights."
        )

    # We will specify derived_from later in the quantizer.
    # We use bias data type to imply the accumulation data type for the output.
    if bias is not None:
        bias = DerivedQuantizationSpec(
            derived_from=None,
            derive_qparams_fn=derive_bias_qparams_fn,
            dtype=bias,
        )

    if qt.microscaling in qschemes:
        assert len(set(qschemes)) == 1, (
            f"Quantization scheme {qschemes[0]} does not work with {qschemes[1]}"
        )
        return get_microscaling_quantizer(input_activation, weight)

    if weight is not None and weight.qscheme == qt.per_channel_symmetric:
        assert weight.ch_axis == 0, (
            f"Per-channel weight quantization only supports quantizing output "
            "channel dimension (dim=0)."
        )

    if input_activation is not None and input_activation.qscheme == qt.per_channel_symmetric:
        return get_per_channel_act_quantizer(input_activation, output_activation, weight, bias)

    qconfig = QuantizationConfig(input_activation, output_activation, weight, bias)
    qconfig_matmul = QuantizationConfig(input_activation, output_activation, input_activation, None)
    return XNNPACKQuantizer().set_global(qconfig) \
        .set_object_type(torch.ops.aten.matmul.default, qconfig_matmul)


def export_model(
    model: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Dict[str, Any]] = None,
):
    from transformers.utils.import_utils import is_torch_greater_or_equal
    if is_torch_greater_or_equal("2.5"):
        from torch.export import export_for_training

        model = export_for_training(model, args, kwargs, dynamic_shapes=dynamic_shapes).module()
    elif is_torch_greater_or_equal("2.0"):
        from torch._export import capture_pre_autograd_graph

        model = capture_pre_autograd_graph(
            model, args, kwargs, dynamic_shapes=dynamic_shapes
        )
    else:
        raise RuntimeError("Torch version is not supported. Please use torch >= 2.0")
    return model


def prepare_pt2e(model, quantizer, args=None, kwargs=None, dynamic_shapes=None):
    from torch.ao.quantization.pt2e import prepare
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e

    # replace the default implementation of _create_obs_or_fq_from_qspec
    prepare._get_obs_or_fq_map = _get_obs_or_fq_map

    if not isinstance(model, GraphModule):
        model = export_model(model, args, kwargs, dynamic_shapes=dynamic_shapes)

    model = prepare_pt2e(model, quantizer)
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
    # device = assert_and_get_unique_device(module)
    # new_value = value.clone().detach() if isinstance(value, torch.Tensor) \
    #     else torch.tensor(value, device=device)
    new_value = value.clone().detach() if isinstance(value, torch.Tensor) \
        else torch.tensor(value)
    module.register_buffer(attr_name, new_value)
    # Create get_attr with value
    attr_node = graph.create_node("get_attr", attr_name)
    return attr_node


def _replace_observer_with_quantize_dequantize_node_decomposed(
    model: torch.fx.GraphModule,
    node: Node,
    modules: Dict[str, torch.nn.Module],
    output_dtype: str = None
):
    graph = model.graph
    assert modules is not None
    assert isinstance(node.target, str)
    activation_post_process = modules[node.target]
    device = assert_and_get_unique_device(activation_post_process)

    torch_dtype = next(iter(model.parameters())).dtype
    scale = activation_post_process.calculate_qparams().to(torch_dtype)
    input_dtype = activation_post_process.dtype

    orig_fq_users = list(node.users.keys())
    input_node = node.args[0]
    if input_node.op == 'get_attr':
        # Quantize weight and remove the fq module
        param = model.get_parameter(input_node.target)
        param.data = torch.ops.quantized_ops.quantize(
            param.data,
            scale,
            input_dtype,
            activation_post_process.code
        )
        node.replace_all_uses_with(input_node)

        # Annotate weight dtype
        input_node.meta["dtype"] = input_dtype

        # Reshape the scale to match the shape of the output tensor for
        # per-channel weight quantization.
        if scale.ndim == 4:
            scale = scale.view(-1, 1, 1)
        elif scale.ndim == 2:
            scale = scale.view(-1)
    else:
        # Replace fake quant module with a quantize node
        with graph.inserting_before(node):
            qparam_node = create_getattr_from_value(
                model, graph, next(iter(node.users)).name + "_scale_", scale)
            # TODO quantization map can be shared among multiple quantize nodes?
            get_attr_node = create_getattr_from_value(
                model, graph, "code_", activation_post_process.code)
            quantize_op_inputs = [node.args[0], qparam_node, input_dtype, get_attr_node]
            quantized_node = graph.call_function(
                torch.ops.quantized_ops.quantize.default,
                tuple(quantize_op_inputs),
                {}
            )

        # source_fn_stack is used by get_source_partitions to find nodes with a given op
        source_fn_st = quantized_node.meta.setdefault("source_fn_stack", [])
        source_fn_st.append((quantized_node.name, quantized_node.target))

        # Annotate input dtype
        quantized_node.meta["dtype"] = input_dtype

        node.replace_all_uses_with(quantized_node)
    graph.erase_node(node)

    # We don't need to insert dequantize node for bias
    if (
        isinstance(activation_post_process, _DerivedObserverOrFakeQuantize)
        or activation_post_process.qscheme is None
    ):
        return

    for user_node in orig_fq_users:
        if _is_gemm_op(user_node):
            user_node.meta["dtype"] = output_dtype

            # Insert dequantize node before the node that appear the earlist in the graph
            node_index_map = {n: i for i, n in enumerate(graph.nodes)}
            all_user_nodes = sorted(
                user_node.users.keys(),
                key=lambda n: node_index_map.get(n, float('inf'))
            )
            maybe_dq_node = all_user_nodes[0]

            if (
                maybe_dq_node.op != "call_function"
                or maybe_dq_node.target != torch.ops.quantized_ops.dequantize.default
            ):
                # Insert a dequantize node after the gemm operation
                code = get_quantization_map(output_dtype, device)
                with graph.inserting_before(maybe_dq_node):
                    qparam_node = create_getattr_from_value(
                        model, graph, user_node.name + "_scale_", scale)
                    get_attr_node = create_getattr_from_value(model, graph, "code_", code)
                    dq_inputs = [user_node, qparam_node, output_dtype, get_attr_node]
                    dequantized_node = graph.call_function(
                        torch.ops.quantized_ops.dequantize.default,
                        tuple(dq_inputs),
                        {}
                    )

                # source_fn_stack is used by get_source_partitions to find nodes
                # associated with a given op
                source_fn_st = dequantized_node.meta.setdefault("source_fn_stack", [])
                source_fn_st.append((dequantized_node.name, dequantized_node.target))

                # We need to save orig users before updating users because
                # the list of users will change as we update users
                orig_users = list(user_node.users.keys())
                for user in orig_users:
                    if id(user) == id(dequantized_node):
                        continue
                    user.replace_input_with(user_node, dequantized_node)
            else:
                # Update the scale if a dequantize node already exists
                qparam_node = maybe_dq_node.args[1]
                buffer = model.get_buffer(qparam_node.target)
                model.register_buffer(qparam_node.target, scale * buffer)
        else:
            # Insert a dequantize node after the quantize node
            with graph.inserting_before(quantized_node.next):
                qparam_node = create_getattr_from_value(
                    model, graph, user_node.name + "_scale_", scale)
                dq_inputs = [quantized_node, qparam_node, None, None]
                dequantized_node = graph.call_function(
                    torch.ops.quantized_ops.dequantize.default,
                    tuple(dq_inputs),
                    {}
                )

            source_fn_st = dequantized_node.meta.setdefault("source_fn_stack", [])
            source_fn_st.append((dequantized_node.name, dequantized_node.target))

            user_node.replace_input_with(quantized_node, dequantized_node)


MX_OP_MAPPING = {
    torch.ops.aten.conv2d.default: torch.ops.quantized_ops.conv2d_mx.default,
    torch.ops.aten.linear.default: torch.ops.quantized_ops.linear_mx.default,
    torch.ops.aten.matmul.default: torch.ops.quantized_ops.matmul_mx.default,
}


def _replace_observer_with_quantize_mx_node_decomposed(
    model: torch.fx.GraphModule,
    node: Node,
    modules: Dict[str, torch.nn.Module],
):
    graph = model.graph
    assert modules is not None
    assert isinstance(node.target, str)
    activation_post_process = modules[node.target]

    input_node = node.args[0]

    node_to_quantize = input_node

    if activation_post_process.outlier_threshold is not None and input_node.op != "get_attr":
        with graph.inserting_before(node):
            filter_node = graph.call_function(
                torch.ops.quantized_ops.filter_outlier.default,
                (input_node, activation_post_process.outlier_threshold),
                {}
            )
            node_to_quantize = graph.call_function(
                operator.getitem,
                (filter_node, 0),
                {}
            )
            csr_data_node = graph.call_function(
                operator.getitem,
                (filter_node, 1),
                {}
            )
            indices_node = graph.call_function(
                operator.getitem,
                (filter_node, 2),
                {}
            )
            indptr_node = graph.call_function(
                operator.getitem,
                (filter_node, 3),
                {}
            )

    device = assert_and_get_unique_device(activation_post_process)
    code = get_quantization_map(activation_post_process.dtype, device)

    if isinstance(code, tuple):
        numbers = re.findall(r'\d+', activation_post_process.dtype)
        N = int(numbers[0])
        activation_post_process.dtype = f"int{N}"

        indices, values = code

        activation_post_process.code = indices

        midpoints = (values[:-1] + values[1:]) / 2
        with graph.inserting_before(node):
            dequant_code = create_getattr_from_value(model, graph, "code_", values)
            quant_code = create_getattr_from_value(model, graph, "code_", midpoints)

        if len(numbers) > 1:
            dequant_code.meta["dtype"] = f"int{numbers[1]}"
    else:
        dequant_code, quant_code = None, None

    if input_node.op == 'get_attr':
        # quantize model parameter and remove the fq module
        param = model.get_parameter(input_node.target)
        scale = torch.ops.quantized_ops.calculate_mx_qparam(
            param.data,
            activation_post_process.ch_axis,
            activation_post_process.quant_max,
            activation_post_process.block_size,
            activation_post_process.force_scale_power_of_two,
            activation_post_process.scale_code,
        )

        weight = torch.ops.quantized_ops.quantize(
            param.data,
            scale,
            activation_post_process.dtype,
            activation_post_process.code,
            activation_post_process.block_size,
        )

        with graph.inserting_before(node):
            scale_node = create_getattr_from_value(
                model, graph, input_node.name + "_scale_", scale)
            quantized_node = create_getattr_from_value(
                model, graph, input_node.name + "_quantized_", weight)
    elif activation_post_process.ch_axis == -2:
        with model.graph.inserting_before(node):
            calculate_qparam_op_inputs = [
                input_node,
                activation_post_process.ch_axis,
                activation_post_process.quant_max,
                activation_post_process.block_size,
                activation_post_process.force_scale_power_of_two,
            ]

            if activation_post_process.scale_code is not None:
                get_attr_node = create_getattr_from_value(
                    model, model.graph, "code_", activation_post_process.scale_code)
                calculate_qparam_op_inputs.append(get_attr_node)

            scale_node = model.graph.call_function(
                torch.ops.quantized_ops.calculate_mx_qparam.default,
                tuple(calculate_qparam_op_inputs),
                {}
            )

            if activation_post_process.scale_dtype is not None:
                scale_node.meta["dtype"] = activation_post_process.scale_dtype

            get_attr_node = create_getattr_from_value(
                model, model.graph, "code_", activation_post_process.code,
            )
            quantize_op_inputs = [
                input_node,
                scale_node,
                activation_post_process.dtype,
                get_attr_node,
                activation_post_process.block_size,
                quant_code,
            ]
            quantized_node = model.graph.call_function(
                torch.ops.quantized_ops.quantize.default,
                tuple(quantize_op_inputs),
                {}
            )

        source_fn_st = quantized_node.meta.setdefault("source_fn_stack", [])
        source_fn_st.append((quantized_node.name, quantized_node.target))
    else:
        with model.graph.inserting_before(node):
            get_attr_node = create_getattr_from_value(
                model, model.graph, "code_", activation_post_process.code,
            )

            scale_code_node = None
            if activation_post_process.scale_code is not None:
                scale_code_node = create_getattr_from_value(
                    model, model.graph, "code_", activation_post_process.scale_code)

            quantize_mx_inputs = [
                node_to_quantize,
                activation_post_process.ch_axis,
                activation_post_process.quant_max,
                activation_post_process.block_size,
                activation_post_process.dtype,
                get_attr_node,
                activation_post_process.force_scale_power_of_two,
                scale_code_node,
                quant_code,
            ]

            quantize_mx_node = model.graph.call_function(
                torch.ops.quantized_ops.quantize_mx.default,
                tuple(quantize_mx_inputs),
                {}
            )
            scale_node = graph.create_node(
                "call_function",
                operator.getitem,
                (quantize_mx_node, 0),
                {}
            )
            quantized_node = graph.create_node(
                "call_function",
                operator.getitem,
                (quantize_mx_node, 1),
                {}
            )

        quantize_mx_node.meta["dtype"] = (
            ("fp8_e8m0" if activation_post_process.force_scale_power_of_two
                else activation_post_process.scale_dtype),
            activation_post_process.dtype,
        )

        source_fn_st = quantize_mx_node.meta.setdefault("source_fn_stack", [])
        source_fn_st.append((quantize_mx_node.name, quantize_mx_node.target))

    quantized_node.meta["dtype"] = activation_post_process.dtype

    if activation_post_process.force_scale_power_of_two:
        scale_node.meta["dtype"] = "fp8_e8m0"
    elif activation_post_process.scale_dtype is not None:
        scale_node.meta["dtype"] = activation_post_process.scale_dtype

    orig_fq_users = list(node.users.keys())

    node.replace_all_uses_with(quantized_node)
    graph.erase_node(node)

    for user in orig_fq_users:
        # Keep the original nodes for other users
        kwarg1, kwarg2 = dequant_code, scale_node

        # Skip device alignment node
        if user.target == torch.Tensor.to:
            user_device = user.args[1]

            with graph.inserting_before(user):
                if kwarg1 is not None:
                    kwarg1 = graph.call_function(
                        torch.Tensor.to, (dequant_code, user_device)
                    )

                kwarg2 = graph.call_function(
                    torch.Tensor.to, (scale_node, user_device)
                )

            user = next(iter(user.users))

        kwargs = OrderedDict(user.kwargs)
        kwargs.setdefault("block_size", activation_post_process.block_size)

        if input_node.op == 'get_attr' or id(quantized_node) == id(user.args[1]):
            kwargs.setdefault("weight_code", kwarg1)
            kwargs.setdefault("weight_scale", kwarg2)
        else:
            kwargs.setdefault("input_code", kwarg1)
            kwargs.setdefault("input_scale", kwarg2)

        # Sort kwargs so that they can be accessed sequentially during MHA splitting
        order = ["input_scale", "weight_scale", "block_size", "input_code", "weight_code"]
        kwargs = OrderedDict((key, kwargs[key]) for key in order if key in kwargs)

        # Replace the node with its MX counterpart
        if user.target in MX_OP_MAPPING:
            with graph.inserting_before(user):
                mx_op_node = graph.call_function(
                    MX_OP_MAPPING[user.target], user.args, kwargs
                )

            user.replace_all_uses_with(mx_op_node)
            graph.erase_node(user)

            mx_op_node.meta = user.meta

            source_fn_st = mx_op_node.meta.setdefault("source_fn_stack", [])
            target = source_fn_st[-1][1] if len(source_fn_st) > 0 else mx_op_node.target
            source_fn_st.append((mx_op_node.name, target))
        elif user.target in MX_OP_MAPPING.values():
            mx_op_node = user
            mx_op_node.kwargs = kwargs
        elif user.target == torch.ops.quantized_ops.spmm_csr.default:
            user.args = user.args[:-1] + (quantized_node,)
        else:
            raise RuntimeError(
                f"Unsupported user node {user.target} for quantization, "
                f"expected one of {list(MX_OP_MAPPING.keys())}"
            )

        if activation_post_process.outlier_threshold is not None and input_node.op != "get_attr":
            # For now only support linear layers
            assert mx_op_node.target in [
                torch.ops.aten.linear.default,
                torch.ops.quantized_ops.linear_mx.default,
            ], (
                f"Only torch.nn.Linear is supported for outlier suppresion, got {user.target}"
            )

            weight_node = mx_op_node.args[1]

            with graph.inserting_before(user):
                spmm_node = graph.call_function(
                    torch.ops.quantized_ops.spmm_csr.default,
                    (csr_data_node, indices_node, indptr_node, weight_node),
                    {
                        "B_scale": scale_node,
                        "B_code": dequant_code,
                        "block_size": activation_post_process.block_size,
                    },
                )

                add_node = graph.call_function(
                    torch.ops.aten.add.Tensor,
                    (spmm_node, mx_op_node),
                    {},
                )

            mx_op_node.replace_all_uses_with(add_node)
            add_node.replace_input_with(add_node, mx_op_node)


def _eliminate_dequantize_with_no_effect(model: GraphModule):
    for node in model.graph.nodes:
        if node.target != torch.ops.quantized_ops.dequantize.default:
            continue

        scale_node = node.args[1]
        scale = model.get_buffer(scale_node.target)
        if torch.any(scale != 1):
            continue

        # During integer quantization, the dequantize node also perform a
        # quantization to the output dtype
        if node.args[2] is not None:
            continue

        node.replace_all_uses_with(node.args[0])
        model.graph.erase_node(node)

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()

    return model


def fuse_quantize_dequantize_with_previous_op(model: GraphModule):
    """
    Move quantize and dequantize nodes up the graph and place them after the
    previous operation (e.g. matmul, conv2d, etc.), so that the quantize and
    dequantize can be fused with the previous operation.
    """
    graph = model.graph

    def find_prev_op_and_move_node(node, prev_node=None):
        nodes_on_path = []
        prev_node = node.args[0] if prev_node is None else prev_node
        while len(prev_node.users) == 1:
            # If there are multiple input nodes, trace each path separately
            if prev_node.target in [
                torch.ops.aten.stack.default, torch.ops.aten.cat.default
            ]:
                nodes_on_path.append(prev_node)
                for arg in prev_node.all_input_nodes:
                    nodes_on_path.extend(find_prev_op_and_move_node(node, arg))
                return nodes_on_path

            # stack and cat are handled above, so we can safely assume that
            # they won't appear here and there is only one input node
            if (
                not _is_nop(prev_node)
                and not _is_reshape_op(prev_node)
                and not _is_indexing_or_concatenation_op(prev_node)
                and prev_node.target not in [
                    torch.ops.aten.expand.default,
                    torch.ops.aten.repeat.default,
                ]
            ):
                break

            assert len(prev_node.all_input_nodes) == 1

            nodes_on_path.append(prev_node)
            prev_node = prev_node.args[0]

        # Check if the quantize or dq node can be moved. In the case of a recursive
        # call, nodes_on_path could be empty, so we need to check if prev_node is
        # the same as the quantize or dq node.
        if id(prev_node) == id(node.args[0]):
            return None

        user = next(iter(prev_node.users))
        with model.graph.inserting_before(user):
            qparam_node = graph.node_copy(node.args[1])
            get_attr_node = graph.node_copy(node.args[3])
            quantize_op_inputs = [prev_node, qparam_node, node.args[2], get_attr_node]
            new_node = graph.call_function(node.target, tuple(quantize_op_inputs), {})

        user.replace_input_with(prev_node, new_node)

        # Copy meta data from the original node, specifically dtype and source_fn_stack
        new_node.meta = {
            k: copy.deepcopy(v) if k != 'val' else v.clone()
            for k, v in node.meta.items()
        }

        # Update source_fn_stack with new node name
        source_fn_stack = new_node.meta.setdefault("source_fn_stack", [])
        target = source_fn_stack[-1][1] if len(source_fn_stack) > 0 else new_node.target
        source_fn_stack.append((new_node.name, target))

        if hasattr(node, "shape"):
            new_node.shape = node.shape
            new_node.value = node.value

        return nodes_on_path

    # First, move the dequantize nodes before the stack and view nodes that
    # are inserted during MHA splitting. Then try to move quantize nodes forward
    # to immediately after their previous ops (nodes that are not NOP).
    for node in list(model.graph.nodes):
        if node.target not in [
            torch.ops.quantized_ops.dequantize.default,
            torch.ops.quantized_ops.quantize.default
        ]:
            continue

        output_node = node
        nodes_on_path = find_prev_op_and_move_node(output_node)
        if nodes_on_path is None:
            continue

        # Update dtype annotation for all nodes on the path.
        is_dequantize = output_node.target == torch.ops.quantized_ops.dequantize.default
        for n in nodes_on_path:
            if is_dequantize:
                n.meta.pop("dtype", None)
            else:
                n.meta["dtype"] = output_node.meta.get("dtype", None)

        # Remove the nodes from the graph. This has to be done at the end
        # because we may need to copy and insert the nodes multiple times
        # if there is stack or cat node in the path.
        output_node.replace_all_uses_with(output_node.args[0])
        graph.erase_node(output_node)

    graph.lint()

    graph.eliminate_dead_code()
    model.recompile()

    return model


def fuse_quantize_with_dequantize(model: GraphModule, output_dtype: str = None):
    import operator

    graph = model.graph

    device = assert_and_get_unique_device(model)

    partitions = get_source_partitions(
        model.graph, [operator.add, torch.add, operator.iadd]
    )
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    for partition in partitions:
        add_node = partition.output_nodes[0]
        print(f"Add node: {add_node.name}")

        input_act0 = add_node.args[0]
        dq_scale0 = 1
        if input_act0.target == torch.ops.quantized_ops.dequantize.default:
            dq_scale0 = model._buffers[input_act0.args[1].target].item()

        input_act1 = add_node.args[1]
        dq_scale1 = 1
        if input_act1.target == torch.ops.quantized_ops.dequantize.default:
            dq_scale1 = model._buffers[input_act1.args[1].target].item()

        # If both inputs are not dequantize nodes, skip
        if dq_scale0 == 1 and dq_scale1 == 1:
            continue

        # TODO handle the case where one of the inputs is a dequantize node
        if dq_scale0 == 1 or dq_scale1 == 1:
            print("One of the inputs is not a dequantize node")
            continue

        node_to_keep = input_act0 if dq_scale0 < dq_scale1 else input_act1

        # Add a dequantize node after the add node
        user = next(iter(add_node.users))
        with graph.inserting_before(user):
            new_node = graph.node_copy(
                node_to_keep,
                lambda n: add_node if n == node_to_keep.args[0] else n
            )
        user.replace_input_with(add_node, new_node)

        node_to_keep.replace_all_uses_with(node_to_keep.args[0])
        graph.erase_node(node_to_keep)
        print(f"Move node: {node_to_keep.name} after node: {add_node.name}")

        node_to_remove = input_act1 if node_to_keep == input_act0 else input_act0

        if dq_scale0 == dq_scale1:
            node_to_remove.replace_all_uses_with(node_to_remove.args[0])
            graph.erase_node(node_to_remove)
            print(f"Remove node: {node_to_remove.name}")
        else:
            qparam_node = node_to_remove.args[1]
            new_scale = max(dq_scale0, dq_scale1) / min(dq_scale0, dq_scale1)

            prev_node = node_to_remove.all_input_nodes[0]
            if (
                prev_node.target == torch.ops.quantized_ops.quantize.default and
                len(prev_node.users) == 1
            ):
                # If the previous node is a quantize node, simply update the dequantize scale.
                # It will be later merged with the quantize node.
                model.register_buffer(qparam_node.target, torch.tensor(new_scale, device=device))
                print(f"Update scale for node: {node_to_remove.name} to {new_scale}\n")
                continue
            else:
                # Else insert a quantize node before the dequantize node with 1 / scale
                model.register_buffer(qparam_node.target, torch.tensor(1 / new_scale, device=device))
                print(f"Update scale for node: {node_to_remove.name} to {new_scale}\n")

                with graph.inserting_before(node_to_remove):
                    get_attr_node = create_getattr_from_value(
                        model, graph, "code_", get_quantization_map(output_dtype, device))
                    quantize_op_inputs = [
                        node_to_remove.args[0], qparam_node, output_dtype, get_attr_node
                    ]
                    quantized_node = graph.call_function(
                        torch.ops.quantized_ops.quantize.default,
                        tuple(quantize_op_inputs),
                        {}
                    )
                node_to_remove.replace_all_uses_with(quantized_node)

                graph.erase_node(node_to_remove)

    partitions = get_source_partitions(
        model.graph, [torch.ops.quantized_ops.dequantize.default]
    )
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    for partition in partitions:
        dequantized_node = partition.output_nodes[0]
        dq_scale = model._buffers[dequantized_node.args[1].target].item()
        print(f"Dequantize node: {dequantized_node.name}")

        # Try to merge the dequantize node with the previous quantize node
        prev_node = dequantized_node.all_input_nodes[0]
        if prev_node.target == torch.ops.quantized_ops.quantize.default and len(prev_node.users) == 1:
            qparam_node = prev_node.args[1]
            q_scale = model._buffers[qparam_node.target].item()

            model.register_buffer(qparam_node.target, torch.tensor(q_scale / dq_scale, device=device))
            print(f"New quantize scale: {q_scale / dq_scale}\n")

            # Update quantization data type
            get_attr_node = prev_node.args[3]
            model.register_buffer(get_attr_node.target, get_quantization_map(output_dtype, device))

            prev_node.args = (prev_node.args[0], qparam_node, output_dtype, get_attr_node)

            dequantized_node.replace_all_uses_with(dequantized_node.args[0])
            graph.erase_node(dequantized_node)
            continue

        match_found = True
        next_node = dequantized_node
        while next_node.target != torch.ops.quantized_ops.quantize.default:
            if len(next_node.users) != 1:
                match_found = False
                break

            if id(next_node) != id(dequantized_node) and next_node.target not in [
                torch.ops.aten.adaptive_avg_pool2d.default,
                torch.ops.aten.max_pool2d.default,
                torch.ops.aten.relu.default,
                torch.ops.aten.relu_.default,
            ]:
                match_found = False
                break

            next_node = next(iter(next_node.users))

        if not match_found:
            continue

        qparam_node = next_node.args[1]
        q_scale = model._buffers[qparam_node.target].item()

        model.register_buffer(qparam_node.target, torch.tensor(q_scale / dq_scale, device=device))
        print(f"Update scale for node: {next_node.name} to {q_scale / dq_scale}\n")

        dequantized_node.replace_all_uses_with(dequantized_node.args[0])
        graph.erase_node(dequantized_node)

    graph.lint()

    graph.eliminate_dead_code()
    model.recompile()

    return model


def convert_pt2e(model: GraphModule, output_dtype: str = None):
    modules = dict(model.named_modules(remove_duplicate=False))

    for node in list(model.graph.nodes):
        if node.op == "call_module":
            mod = _get_module(node, modules)
            assert mod is not None
            if isinstance(mod, torch.ao.quantization.FakeQuantizeBase):
                if mod.qscheme == qt.microscaling:
                    _replace_observer_with_quantize_mx_node_decomposed(model, node, modules)
                else:
                    _replace_observer_with_quantize_dequantize_node_decomposed(
                        model, node, modules, output_dtype)

    _eliminate_dequantize_with_no_effect(model)

    model.graph.lint()
    model.recompile()

    model.delete_all_unused_submodules()

    return model
