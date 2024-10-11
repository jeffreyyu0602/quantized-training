import copy
import itertools
from dataclasses import asdict, replace
from typing import Dict, Tuple, Any, Optional, Callable, List

import torch
from torch import Tensor
from torch._export import capture_pre_autograd_graph
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
    _get_quantization_map,
)
from quantized_training.quantizer.quantizer import QuantizationSpec, DerivedQuantizationSpec
from quantized_training.quantizer.xnnpack_quantizer import XNNPACKQuantizer
from quantized_training.quantizer.xnnpack_quantizer_utils import QuantizationConfig

from .codegen.mapping_utils import _is_nop, _is_gemm_op
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
    outlier_threshold: float = None,
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
        outlier_threshold=outlier_threshold,
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


def prepare_pt2e(model, quantizer, args, kwargs=None, dynamic_shapes=None):
    from torch.ao.quantization.pt2e import prepare
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e

    # HACK monkey patching to replace the default implementation of _create_obs_or_fq_from_qspec
    prepare._get_obs_or_fq_map = _get_obs_or_fq_map

    model = capture_pre_autograd_graph(
        model,
        args=args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
    )

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
    device = assert_and_get_unique_device(module)
    new_value = value.clone().detach() if isinstance(value, torch.Tensor) \
        else torch.tensor(value, device=device)
    module.register_buffer(attr_name, new_value)
    # Create get_attr with value
    attr_node = graph.create_node("get_attr", attr_name)
    return attr_node


def get_accumulation_dtype(dtype):
    import re
    if re.fullmatch(r'int(\d+)', dtype):
        return 'int32'
    return None


def _replace_observer_with_quantize_dequantize_node_decomposed(
    model: torch.fx.GraphModule,
    node: Node,
    modules: Dict[str, torch.nn.Module],
):
    graph = model.graph
    assert modules is not None
    assert isinstance(node.target, str)
    activation_post_process = modules[node.target]
    device = assert_and_get_unique_device(activation_post_process)

    torch_dtype = next(iter(model.parameters())).dtype
    scale = activation_post_process.calculate_qparams().to(torch_dtype)
    input_dtype = activation_post_process.dtype

    # HACK this logic needs to be improved. Accumulation and output data types are set
    # to match the bias data type. We loop through all nodes, locate any conv2d or linear
    # nodes, and extract their bias data type.
    output_dtype = None
    for n in model.graph.nodes:
        if n.target not in [torch.ops.aten.conv2d.default, torch.ops.aten.linear.default]:
            continue
        bias_n = n.args[2]
        if bias_n.op == 'get_attr':
            output_dtype = bias_n.meta.get("dtype", None)
        elif bias_n.op == 'call_module':
            output_dtype = modules[bias_n.target].dtype
        break

    param_dict = dict(model.named_parameters())
    orig_fq_users = list(node.users.keys())
    input_node = node.args[0]
    if input_node.op == 'get_attr':
        # Quantize weight and remove the fq module
        param = param_dict[input_node.target]
        param.data = torch.ops.quantized_ops.quantize_symmetric(
            param.data,
            scale,
            input_dtype,
            activation_post_process.quant_map
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
            qmap_node = create_getattr_from_value(
                model, graph, "code_", activation_post_process.quant_map)
            quantize_op_inputs = [node.args[0], qparam_node, input_dtype, qmap_node]
            quantized_node = graph.call_function(
                torch.ops.quantized_ops.quantize_symmetric,
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
            # Insert a dequantize node after the gemm operation
            if output_dtype is None:
                output_dtype = get_accumulation_dtype(input_dtype)
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
                or maybe_dq_node.target != torch.ops.quantized_ops.dequantize_symmetric
            ):
                quant_map = _get_quantization_map(output_dtype, device)
                with graph.inserting_before(maybe_dq_node):
                    qparam_node = create_getattr_from_value(
                        model, graph, user_node.name + "_scale_", scale)
                    qmap_node = create_getattr_from_value(model, graph, "code_", quant_map)
                    dq_inputs = [user_node, qparam_node, output_dtype, qmap_node]
                    dequantized_node = graph.call_function(
                        torch.ops.quantized_ops.dequantize_symmetric,
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
                    torch.ops.quantized_ops.dequantize_symmetric,
                    tuple(dq_inputs),
                    {}
                )

            source_fn_st = dequantized_node.meta.setdefault("source_fn_stack", [])
            source_fn_st.append((dequantized_node.name, dequantized_node.target))

            user_node.replace_input_with(quantized_node, dequantized_node)


QUANT_OP_MAPPINGS = {
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

    param_dict = dict(model.named_parameters())
    orig_fq_users = list(node.users.keys())

    qparam_node_inp = None
    qparam_node_wt = None
    input_node = node.args[0]
    if input_node.op == 'get_attr':
        # quantize model parameter and remove fq module
        param = param_dict[input_node.target]
        scale = torch.ops.quantized_ops.calculate_mx_qparam(
            param.data,
            activation_post_process.ch_axis,
            activation_post_process.quant_max,
            activation_post_process.block_size,
            activation_post_process.force_scale_power_of_two,
        )
        with graph.inserting_before(node):
            qparam_node_wt = create_getattr_from_value(
                model, graph, input_node.name + "_scale_", scale)

        param.data = torch.ops.quantized_ops.quantize_symmetric(
            param.data,
            scale,
            activation_post_process.dtype,
            activation_post_process.quant_map,
            activation_post_process.block_size,
        )
        node.replace_all_uses_with(input_node)

        # Annotate weight dtype
        input_node.meta["dtype"] = activation_post_process.dtype

        if activation_post_process.force_scale_power_of_two:
            qparam_node_wt.meta["dtype"] = "e8m0"
    else:
        with model.graph.inserting_before(node):
            calculate_qparam_op_inputs = [
                node.args[0],
                activation_post_process.ch_axis,
                activation_post_process.quant_max,
                activation_post_process.block_size,
                activation_post_process.force_scale_power_of_two,
            ]
            qparam_node_inp = model.graph.call_function(
                torch.ops.quantized_ops.calculate_mx_qparam,
                tuple(calculate_qparam_op_inputs),
                {}
            )
            qmap_node = create_getattr_from_value(
                model, model.graph, "code_", activation_post_process.quant_map)
            quantize_op_inputs = [
                node.args[0],
                qparam_node_inp,
                activation_post_process.dtype,
                qmap_node,
                activation_post_process.block_size,
            ]
            quantized_node = model.graph.call_function(
                torch.ops.quantized_ops.quantize_symmetric,
                tuple(quantize_op_inputs),
                {}
            )

        source_fn_st = quantized_node.meta.setdefault("source_fn_stack", [])
        source_fn_st.append((quantized_node.name, quantized_node.target))

        node.replace_all_uses_with(quantized_node)

        # Annotate input dtype
        quantized_node.meta["dtype"] = activation_post_process.dtype

        if activation_post_process.force_scale_power_of_two:
            qparam_node_inp.meta["dtype"] = "e8m0"
    graph.erase_node(node)

    for user_node in orig_fq_users:
        new_kwargs = dict(user_node.kwargs)
        new_kwargs.setdefault("block_size", activation_post_process.block_size)
        if qparam_node_inp is not None:
            # Handle the second argument of torch.matmul
            if id(quantized_node) == id(user_node.args[1]):
                new_kwargs.setdefault("scale_wt", qparam_node_inp)
            else:
                new_kwargs.setdefault("scale_inp", qparam_node_inp)
        if qparam_node_wt is not None:
            new_kwargs.setdefault("scale_wt", qparam_node_wt)
        # If user_node is a not a mx op, replace the node with its mx counterpart.
        # otherwise, replace one of its input with the quantized node
        if user_node.target in QUANT_OP_MAPPINGS:
            with graph.inserting_before(user_node):
                new_node = graph.call_function(
                    QUANT_OP_MAPPINGS[user_node.target],
                    user_node.args,
                    new_kwargs,
                )
            new_node.meta = user_node.meta
            source_fn_st = new_node.meta.setdefault("source_fn_stack", [])
            source_fn_st.append((new_node.name, new_node.target))
            user_node.replace_all_uses_with(new_node)
            graph.erase_node(user_node)
        elif user_node.target in QUANT_OP_MAPPINGS.values():
            user_node.kwargs = new_kwargs


def _eliminate_dequantize_with_no_effect(model: GraphModule):
    partitions = get_source_partitions(
        model.graph, [torch.ops.quantized_ops.dequantize_symmetric]
    )
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    for partition in partitions:
        dequantized_node = partition.output_nodes[0]

        scale_node = dequantized_node.args[1]
        scale = model.get_buffer(scale_node.target)
        if torch.any(scale != 1):
            continue

        dtype = dequantized_node.args[2]
        if dtype is not None:
            continue

        dequantized_node.replace_all_uses_with(dequantized_node.args[0])
        model.graph.erase_node(dequantized_node)


def _fuse_quantize_dequantize_with_previous_op(model: GraphModule):
    graph = model.graph

    def find_node_and_insert(node, prev_node=None, scale=None):
        nodes_on_path = []
        prev_node = node.args[0] if prev_node is None else prev_node
        while len(prev_node.users) == 1:
            if _is_nop(prev_node) or prev_node.target in [
                torch.ops.aten.permute.default, torch.ops.aten.transpose.int,
            ]:
                nodes_on_path.append(prev_node)
                prev_node = prev_node.args[0]
            # Our accelerator doesn't support fusing quantize with dequantize
            # elif prev_node.target == torch.ops.quantized_ops.dequantize_symmetric:
            #     # Update scale
            #     scale = model.get_buffer(prev_node.args[1].target)
            #     node_to_remove = prev_node
            #     prev_node = prev_node.args[0]
            #     # Remove dequantize node since it has no effect
            #     node_to_remove.replace_all_uses_with(prev_node)
            #     graph.erase_node(node_to_remove)
            elif prev_node.target in [torch.ops.aten.stack.default, torch.ops.aten.cat.default]:
                # If there is a split, trace each branch separately
                nodes_on_path.append(prev_node)
                for arg in prev_node.all_input_nodes:
                    nodes_on_path.extend(find_node_and_insert(node, arg, scale))
                return nodes_on_path
            else:
                break

        if id(prev_node) == id(node.args[0]):
            return None

        user = next(iter(prev_node.users))
        with model.graph.inserting_before(user):
            # q_scale = model.get_buffer(orig_args[1].target)
            # if scale is not None:
            #     q_scale = q_scale / scale
            # qparam_node = create_getattr_from_value(
            #     model, graph, orig_args[1].name.rsplit('_', 1)[0] + '_', q_scale)
            qparam_node = graph.node_copy(node.args[1])
            qmap_node = graph.node_copy(node.args[3])
            quantize_op_inputs = [prev_node, qparam_node, node.args[2], qmap_node]
            new_node = graph.call_function(node.target, tuple(quantize_op_inputs), {})

        user.replace_input_with(prev_node, new_node)
        new_node.meta = {k: copy.deepcopy(v) for k, v in node.meta.items() if k != 'val'}
        source_fn_st = new_node.meta.setdefault("source_fn_stack", [])
        source_fn_st.append((new_node.name, new_node.target))

        return nodes_on_path

    # Switch the order of dequantize nodes with stack and view nodes inserted
    # during MHA splitting.
    partitions = get_source_partitions(
        model.graph, [torch.ops.quantized_ops.dequantize_symmetric]
    )
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    for partition in partitions:
        output_node = partition.output_nodes[0]
        nodes = find_node_and_insert(output_node)
        if nodes is None:
            continue

        dq_inputs = list(output_node.args)

        for n in nodes:
            del n.meta["dtype"]

        output_node.replace_all_uses_with(dq_inputs[0])

        graph.erase_node(output_node)
        graph.erase_node(dq_inputs[1])
        graph.erase_node(dq_inputs[3])

    partitions = get_source_partitions(
        model.graph, [torch.ops.quantized_ops.quantize_symmetric]
    )
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    for partition in partitions:
        output_node = partition.output_nodes[0]
        nodes = find_node_and_insert(output_node)
        if nodes is None:
            continue

        quantize_op_inputs = list(output_node.args)

        for n in nodes:
            n.meta["dtype"] = quantize_op_inputs[2]

        output_node.replace_all_uses_with(quantize_op_inputs[0])

        graph.erase_node(output_node)
        graph.erase_node(quantize_op_inputs[1])
        graph.erase_node(quantize_op_inputs[3])


def propagate_fake_tensor(model: GraphModule, example_inputs: Tuple[torch.Tensor]):
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp
    from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
    with FakeTensorMode(allow_non_fake_inputs=True) as fake_tensor_mode:
        def to_fake_tensor(x):
            if isinstance(x, torch.Tensor) and not isinstance(x, FakeTensor):
                return fake_tensor_mode.from_tensor(x)
            return x
        example_inputs = tuple(map(to_fake_tensor, example_inputs))
        FakeTensorProp(model, fake_tensor_mode).propagate(*example_inputs)
    for node in model.graph.nodes:
        if 'val' not in node.meta:
            print(f"Node {node} does not have a val attribute")


def convert_pt2e(model: GraphModule):
    modules = dict(model.named_modules(remove_duplicate=False))

    for node in list(model.graph.nodes):
        if node.op == "call_module":
            mod = _get_module(node, modules)
            assert mod is not None
            if isinstance(mod, torch.ao.quantization.FakeQuantizeBase):
                if mod.qscheme == qt.microscaling:
                    _replace_observer_with_quantize_mx_node_decomposed(model, node, modules)
                else:
                    _replace_observer_with_quantize_dequantize_node_decomposed(model, node, modules)

    _eliminate_dequantize_with_no_effect(model)
    _fuse_quantize_dequantize_with_previous_op(model)

    model.graph.lint()
    model.recompile()
    return model
