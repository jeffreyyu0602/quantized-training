import itertools
import operator
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional

import torch
from torch.ao.quantization.fx.utils import (
    get_new_attr_name_with_prefix,
    assert_and_get_unique_device,
)
from torch.ao.quantization.pt2e.export_utils import _WrapperModule
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.quantizer import QuantizationAnnotation

from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
)
from torch.fx import Node
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap,
)
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from .quantizer import QuantizationSpec, DerivedQuantizationSpec


# In the absence of better name, just winging it with QuantizationConfig
@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    input_activation: Optional[QuantizationSpec]
    output_activation: Optional[QuantizationSpec]
    weight: Optional[QuantizationSpec]
    bias: Optional[QuantizationSpec]
    # TODO: remove, since we can use observer_or_fake_quant_ctr to express this
    is_qat: bool = False


OperatorPatternType = List[Callable]
OperatorPatternType.__module__ = (
    "torch.ao.quantization.quantizer.xnnpack_quantizer_utils"
)

AnnotatorType = Callable[
    [
        torch.fx.GraphModule,
        Optional[QuantizationConfig],
        Optional[Callable[[Node], bool]],
    ],
    Optional[List[List[Node]]],
]
OP_TO_ANNOTATOR: Dict[str, AnnotatorType] = {}


def register_annotator(op: str):
    def decorator(annotator: AnnotatorType):
        OP_TO_ANNOTATOR[op] = annotator

    return decorator


def _is_annotated(nodes: List[Node]):
    """
    Given a list of nodes (that represents an operator pattern),
    check if any of the node is annotated, return True if any of the node
    is annotated, otherwise return False
    """
    annotated = False
    for node in nodes:
        annotated = annotated or (
            "quantization_annotation" in node.meta
            and node.meta["quantization_annotation"]._annotated
        )
    return annotated


def _mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if node is not None:
            if "quantization_annotation" not in node.meta:
                node.meta["quantization_annotation"] = QuantizationAnnotation()
            node.meta["quantization_annotation"]._annotated = True


@register_annotator("linear")
def _annotate_linear(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    input_act_qspec = quantization_config.input_activation
    output_act_qspec = quantization_config.output_activation
    weight_qspec = quantization_config.weight
    bias_qspec = quantization_config.bias
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.linear.default:
            continue
        if filter_fn and not filter_fn(node):
            continue
        act_node = node.args[0]
        weight_node = node.args[1]
        bias_node = None
        if len(node.args) > 2:
            bias_node = node.args[2]

        if isinstance(quantization_config.bias, DerivedQuantizationSpec):
            bias_qspec = replace(
                quantization_config.bias,
                derived_from=[(act_node, node), (weight_node, node)],
            )

        if _is_annotated([node]) is False:  # type: ignore[list-item]
            _annotate_input_qspec_map(
                node,
                act_node,
                input_act_qspec,
            )
            _annotate_input_qspec_map(
                node,
                weight_node,
                weight_qspec,
            )
            nodes_to_mark_annotated = [node, weight_node]
            if bias_node:
                _annotate_input_qspec_map(
                    node,
                    bias_node,
                    bias_qspec,
                )
                nodes_to_mark_annotated.append(bias_node)
            _annotate_output_qspec(node, output_act_qspec)
            _mark_nodes_as_annotated(nodes_to_mark_annotated)
            annotated_partitions.append(nodes_to_mark_annotated)

    return annotated_partitions


@register_annotator("conv")
def _annotate_conv(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for n in gm.graph.nodes:
        if n.op != "call_function" or n.target not in [
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv2d.default,
        ]:
            continue
        conv_node = n

        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = quantization_config.input_activation

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = quantization_config.weight

        # adding weight node to the partition as well
        partition = [conv_node, conv_node.args[1]]

        bias_qspec = quantization_config.bias
        if isinstance(bias_qspec, DerivedQuantizationSpec):
            bias_qspec = replace(
                quantization_config.bias,
                derived_from=[(input_act, n), (weight, n)],
            )

        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = bias_qspec
            partition.append(bias)

        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("matmul")
def _annotate_matmul(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    input_act_qspec = quantization_config.input_activation
    output_act_qspec = quantization_config.output_activation
    weight_qspec = quantization_config.weight
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.matmul.default:
            continue
        if filter_fn and not filter_fn(node):
            continue
        if _is_annotated([node]):
            continue

        input_qspec_map = {}
        input_act0 = node.args[0]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = input_act_qspec

        # We use weight_qspec for the second input to differentiate
        # the two inputs of torch.matmul
        input_act1 = node.args[1]
        if isinstance(input_act1, Node):
            input_qspec_map[input_act1] = weight_qspec

        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )
        annotated_partitions.append(node)
    return annotated_partitions


@register_annotator("residual")
def _annotate_residual(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    node_order = {node: i for i, node in enumerate(gm.graph.nodes)}
    add_partitions = get_source_partitions(
        gm.graph, [operator.add, torch.add, operator.iadd]
    )
    add_partitions = list(itertools.chain.from_iterable(add_partitions.values()))
    annotated_partitions = []
    for add_partition in add_partitions:
        annotated_partitions.append(add_partition.nodes)
        add_node = add_partition.output_nodes[0]
        if filter_fn and not filter_fn(add_node):
            continue
        if _is_annotated([add_node]):
            continue

        input_act_qspec = quantization_config.input_activation
        output_act_qspec = quantization_config.output_activation

        input_act0 = add_node.args[0]
        input_act1 = add_node.args[1]

        if (
            not isinstance(input_act0, Node)
            or not isinstance(input_act1, Node)
            or input_act0.op == "get_attr"
            or input_act1.op == "get_attr"
            or input_act0.meta['val'].shape != input_act1.meta['val'].shape
        ):
            continue

        node_to_quantize = input_act0 if node_order[input_act0] < node_order[input_act1] else input_act1
        input_qspec_map = {node_to_quantize: input_act_qspec}

        add_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    return annotated_partitions


@register_annotator("add")
def _annotate_add(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    add_partitions = get_source_partitions(
        gm.graph, [operator.add, torch.add, operator.iadd], filter_fn
    )
    add_partitions = list(itertools.chain.from_iterable(add_partitions.values()))
    annotated_partitions = []
    for add_partition in add_partitions:
        annotated_partitions.append(add_partition.nodes)
        add_node = add_partition.output_nodes[0]
        if _is_annotated([add_node]):
            continue

        input_act_qspec = quantization_config.input_activation
        output_act_qspec = quantization_config.output_activation

        input_qspec_map = {}
        input_act0 = add_node.args[0]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = add_node.args[1]
        if isinstance(input_act1, Node):
            input_qspec_map[input_act1] = input_act_qspec

        add_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    return annotated_partitions


@register_annotator("mul")
def _annotate_mul(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    mul_partitions = get_source_partitions(
        gm.graph, ["mul", "mul_", operator.mul, torch.mul, operator.imul], filter_fn
    )
    mul_partitions = list(itertools.chain.from_iterable(mul_partitions.values()))
    annotated_partitions = []
    for mul_partition in mul_partitions:
        annotated_partitions.append(mul_partition.nodes)
        mul_node = mul_partition.output_nodes[0]
        if _is_annotated([mul_node]):
            continue

        input_act_qspec = quantization_config.input_activation
        output_act_qspec = quantization_config.output_activation

        input_qspec_map = {}
        input_act0 = mul_node.args[0]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = mul_node.args[1]
        if isinstance(input_act1, Node):
            input_qspec_map[input_act1] = input_act_qspec

        mul_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    return annotated_partitions


# TODO: make the list of ops customizable
def _convert_scalars_to_attrs(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    device = assert_and_get_unique_device(model)
    model_dtype = next(model.parameters()).dtype
    for n in model.graph.nodes:
        if n.op != "call_function" or n.target not in [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.div.Tensor,
        ]:
            continue
        args = list(n.args)
        new_args = []
        for i in range(len(args)):
            if isinstance(args[i], torch.fx.Node):
                new_args.append(args[i])
                continue
            prefix = "_tensor_constant_"
            get_new_attr_name = get_new_attr_name_with_prefix(prefix)
            tensor_constant_name = get_new_attr_name(model)
            float_tensor = torch.tensor(float(args[i]), dtype=model_dtype, device=device)
            model.register_buffer(tensor_constant_name, float_tensor)
            fake_mode = n.meta["val"].fake_mode
            with model.graph.inserting_before(n):
                get_attr_node = model.graph.create_node(
                    "get_attr", tensor_constant_name, (), {}
                )
                get_attr_node.meta["val"] = fake_mode.from_tensor(
                    float_tensor, static_shapes=True
                )
                new_args.append(get_attr_node)
        n.args = tuple(new_args)
    model.recompile()
    return model
