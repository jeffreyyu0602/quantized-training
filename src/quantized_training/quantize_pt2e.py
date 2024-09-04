import copy
import itertools
import math
from dataclasses import asdict, replace
from typing import Dict, Tuple, Union, Any, Optional, Callable

import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.fx.utils import assert_and_get_unique_device
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
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
from quantized_training.codegen.mapping import _decompose_node
from quantized_training.export_utils import _allow_exported_model_train_eval
from quantized_training.fake_quantize import FusedAmaxObsFakeQuantize, get_fake_quant_fn
from quantized_training.quantizer.quantizer import QuantizationSpec
from quantized_training.quantizer.xnnpack_quantizer import XNNPACKQuantizer
from quantized_training.quantizer.xnnpack_quantizer_utils import QuantizationConfig

from .decomposed import quantized_decomposed_lib
from .mx_utils import _reshape_to_blocks, _shared_exponents


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
        .set_operator_type(torch.ops.aten.matmul.default, qconfig_matmul)
    )


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

    return (XNNPACKQuantizer()
        .set_module_type(torch.nn.Conv2d, qconfig_conv2d)
        .set_module_type(torch.nn.Linear, qconfig_linear)
        .set_operator_type(torch.ops.aten.matmul.default, qconfig_matmul)
    )


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

    if qt.microscaling in qschemes:
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

    model = capture_pre_autograd_graph(
        model,
        args=args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
    )

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


QUANTIZATION_DTYPES = {
    "int8": {
        "output": "int24",
        "bias": "int24",
    }
}


def _get_quantization_map(dtype, device):
    values = torch.arange(2 ** 16, dtype=torch.int16).view(torch.bfloat16)
    if dtype is not None:
        fq_fn = get_fake_quant_fn(dtype)
        values = fq_fn(values)
    return values.to(device)


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

    input_dtype = activation_post_process.dtype
    output_dtype = None
    bias_dtype = None
    if input_dtype in QUANTIZATION_DTYPES:
        output_dtype, bias_dtype = QUANTIZATION_DTYPES[input_dtype].values()

    param = next(iter(model.parameters()))
    scale = activation_post_process.scale.to(param.dtype)

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
            # TODO quantization map node can be shared among multiple quantize nodes
            quant_map_node = create_getattr_from_value(
                model, graph, "quant_map_", activation_post_process.quant_map)
            quantize_op_inputs = [node.args[0], qparam_node, input_dtype, quant_map_node]
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

    # Insert a dequantize node after each user
    for user_node in orig_fq_users:
        user_node.meta["dtype"] = output_dtype
        maybe_dq_node = next(iter(user_node.users))
        if (
            maybe_dq_node.op != "call_function"
            or maybe_dq_node.target != torch.ops.quantized_ops.dequantize_symmetric
        ):
            quant_map = _get_quantization_map(output_dtype, device)
            with graph.inserting_before(maybe_dq_node):
                qparam_node = create_getattr_from_value(
                    model, graph, user_node.name + "_scale_", scale)
                quant_map_node = create_getattr_from_value(model, graph, "quant_map_", quant_map)
                dq_inputs = [user_node, qparam_node, output_dtype, quant_map_node]
                dequantized_node = graph.call_function(
                    torch.ops.quantized_ops.dequantize_symmetric,
                    tuple(dq_inputs),
                    {}
                )

            # source_fn_stack is used by get_source_partitions to find nodes
            # associated with a given op
            source_fn_st = dequantized_node.meta.setdefault("source_fn_stack", [])
            source_fn_st.append((dequantized_node.name, dequantized_node.target))

            # We need to save orig users before updating uses because
            # the list of users will change as we update uses
            orig_users = list(user_node.users.keys())
            for user in orig_users:
                if id(user) == id(dequantized_node):
                    continue
                user.replace_input_with(user_node, dequantized_node)
        else:
            # Update the scale if a dequantize node already exists
            qparam_node = maybe_dq_node.args[1]
            buffer = model.get_buffer(qparam_node.target)
            model.register_buffer(qparam_node.target, scale.mul_(buffer))

        if user_node.op != 'call_function' or user_node.target not in [
            torch.ops.aten.conv2d.default,
            torch.ops.aten.linear.default,
        ]:
            continue

        if (bias_node := user_node.args[2]) is None or bias_dtype is None:
            continue

        # Quantize bias using the product of input activation scale and weight scale.
        # Save the original bias value because we may need to update it twice with
        # both the input scale and the weight scale.
        param = param_dict[bias_node.target]
        if (bias_value := bias_node.meta.get("param_data")) is None:
            bias_value = param.data.clone()
            bias_node.meta["param_data"] = bias_value

        quant_map = _get_quantization_map(bias_dtype, device)
        param.data = torch.ops.quantized_ops.quantize_symmetric(
            bias_value, scale.flatten(), bias_dtype, quant_map)

        bias_node.meta["dtype"] = bias_dtype


QUANT_OP_MAPPINGS = {
    torch.ops.aten.conv2d.default: torch.ops.quantized_ops.conv2d_mx,
    torch.ops.aten.linear.default: torch.ops.quantized_ops.linear_mx,
    torch.ops.aten.matmul.default: torch.ops.quantized_ops.matmul_mx,
}


def _calculate_mx_qparam(
    input: torch.Tensor,
    quant_max: float,
    axes: Union[int, Tuple[int]],
    block_size: int = 32,
) -> torch.Tensor:
    axes = [axes] if type(axes) == int else axes
    axes = [x + input.ndim if x < 0 else x for x in axes]

    # Perform tiling to the hardware vector size
    if block_size > 0:
        reshaped, axes, orig_shape, padded_shape = _reshape_to_blocks(
            input, axes, block_size
        )

    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

    # Get shared exponents
    shared_exp = _shared_exponents(
        reshaped, method="max", axes=shared_exp_axes, ebits=0,
    )

    # Offset the max exponent by the largest representable exponent
    # in the element data format
    shared_exp = shared_exp - math.floor(math.log2(quant_max))

    for axis in reversed(axes):
        # Remove extra dimension
        shared_exp = torch.squeeze(shared_exp, dim=axis + 1)

    return (2 ** shared_exp).to(input.dtype)


def _calculate_mx_padding(A, axes, block_size):
    if axes is None:
        raise Exception(
            "axes required in order to determine which "
            "dimension toapply block size to"
        )
    if block_size == 0:
        raise Exception("block_size == 0 in _reshape_to_blocks")

    # Fix axes to be positive and sort them
    axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
    assert all(x >= 0 for x in axes)
    axes = sorted(axes)

    # Add extra dimension for tiles
    for i in range(len(axes)):
        axes[i] += i  # Shift axes due to added dimensions
        A = torch.unsqueeze(A, dim=axes[i] + 1)

    # Pad to block_size
    orig_shape = A.size()
    pad = []
    for i in range(len(orig_shape)):
        pad += [0, 0]

    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if isinstance(pre_pad_size, torch.Tensor):
            pre_pad_size = int(pre_pad_size.value)
        # Don't pad if the axis is short enough to fit inside one tile
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True

    if do_padding:
        pad = list(reversed(pad))
        A = torch.nn.functional.pad(A, pad, mode="constant")

    def _reshape(shape, reshape_block_size):
        for axis in axes:
            # Reshape to tiles if axis length > reshape_block_size
            if shape[axis] >= reshape_block_size:
                assert shape[axis] % reshape_block_size == 0
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            # Otherwise preserve length and insert a 1 into the shape
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    # Reshape to tiles
    padded_shape = A.size()
    reshape = _reshape(list(padded_shape), block_size)

    return pad, orig_shape, reshape


def _replace_mx_observer_node(
    model: torch.nn.Module,
    node: Node,
    input: torch.Tensor,
    activation_post_process: torch.nn.Module,
) -> torch.Tensor:
    axes = activation_post_process.ch_axis
    axes = [axes] if type(axes) == int else axes
    axes = [x + input.ndim if x < 0 else x for x in axes]

    block_size = activation_post_process.block_size
    pad, orig_shape, padded_shape = _calculate_mx_padding(
        input, axes, block_size
    )
    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes
    emax = math.floor(math.log2(activation_post_process.quant_max))

    @torch._dynamo.assume_constant_result
    def get_quantization_map():
        return activation_post_process.quant_map

    class QuantizeMX(torch.nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor: 
            reshaped = input.view(orig_shape)
            reshaped = torch.nn.functional.pad(reshaped, pad, mode="constant")
            reshaped = reshaped.view(padded_shape)

            amax = torch.amax(torch.abs(reshaped), dim=shared_exp_axes)
            shared_exp = torch.floor(torch.log2(amax + (amax == 0).type(amax.dtype))) - emax
            scale = 2 ** shared_exp
            quant_map = get_quantization_map()
            input = torch.ops.quantized_ops.quantize_symmetric(
                input, scale, activation_post_process.dtype, quant_map, block_size
            )
            return (input, scale)

    gm = capture_pre_autograd_graph(QuantizeMX(), (input,))
    return _decompose_node(model, gm, node)


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

    input_dtype = activation_post_process.dtype

    qparam_node_inp = None
    qparam_node_wt = None
    input_node = node.args[0]
    if input_node.op == 'get_attr':
        # quantize model parameter and remove fq module
        param = param_dict[input_node.target]
        scale = _calculate_mx_qparam(
            param.data,
            activation_post_process.quant_max,
            activation_post_process.ch_axis,
            activation_post_process.block_size,
        )
        with graph.inserting_before(node):
            qparam_node_wt = create_getattr_from_value(
                model, graph, next(iter(node.users)).name + "_scale_", scale)

        param.data = torch.ops.quantized_ops.quantize_symmetric(
            param.data,
            scale,
            input_dtype,
            activation_post_process.quant_map,
            activation_post_process.block_size,
        )
        node.replace_all_uses_with(input_node)

        # annotate weight node dtype
        input_node.meta["dtype"] = input_dtype
    else:
        quantized_node, qparam_node_inp = _replace_mx_observer_node(
            model,
            node,
            input_node.meta['val'],
            activation_post_process,
        )
        quantized_node.meta["dtype"] = input_dtype
    graph.erase_node(node)

    for user_node in orig_fq_users:
        new_kwargs = dict(user_node.kwargs)
        new_kwargs.setdefault("block_size", activation_post_process.block_size)
        if qparam_node_inp is not None:
            # handle the second argument of torch.matmul
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
            user_node.replace_all_uses_with(new_node)
            graph.erase_node(user_node)
        elif user_node.target in QUANT_OP_MAPPINGS.values():
            user_node.kwargs = new_kwargs


def _fuse_dequantize_quantize(model: GraphModule):
    partitions = get_source_partitions(
        model.graph, [torch.ops.quantized_ops.dequantize_symmetric]
    )
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    for partition in partitions:
        dequantize_node = partition.output_nodes[0]
        if (
            dequantize_node.op != "call_function"
            or dequantize_node.target != torch.ops.quantized_ops.dequantize_symmetric
        ):
            continue

        # TODO combine dequantize and quantize into a single quantize node

        scale_node = dequantize_node.args[1]
        scale = model.get_buffer(scale_node.target)
        if torch.any(scale != 1):
            continue

        dtype = dequantize_node.args[2]
        if dtype is not None:
            continue

        dequantize_node.replace_all_uses_with(dequantize_node.args[0])
        model.graph.erase_node(dequantize_node)


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

    _fuse_dequantize_quantize(model)

    model.graph.lint()
    model.recompile()
    return model
