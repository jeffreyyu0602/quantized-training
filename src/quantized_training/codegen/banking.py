from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import torch

from .memory import Segment, MemorySpace, _align_size, _get_tensor_size


logger = logging.getLogger(__name__)


class BankingPolicy(Enum):
    """
    Policy for handling tensors that are not explicitly listed in any partition.
    """
    SEPARATE_ALIGNED = auto()
    SEPARATE_UNALIGNED = auto()


@dataclass(frozen=True)
class BankPartition:
    """
    A bank partition groups multiple logical tensor roles into a single
    bank-aligned allocation unit.

    - tensors: logical tensor roles (each role maps to at most one node)
    - max_banks: optional constraint on how many banks this partition may span
    - align: whether to apply bank-size rounding to the aggregate size
    """
    tensors: Tuple[str, ...]
    max_banks: Optional[int] = None
    align: bool = True


@dataclass(frozen=True)
class BankingStrategy:
    """
    A banking strategy is an ordered list of partitions.
    Earlier strategies are assumed to be higher-performance.
    """
    partitions: Tuple[BankPartition, ...]
    unspecified_policy: BankingPolicy = BankingPolicy.SEPARATE_ALIGNED

    def evaluate(self, key_to_node, node, tile_shapes, bank_width, bank_size):
        specified_nodes = set()

        node_to_segment: Dict[Any, Segment] = {}
        current_offset = 0

        # Only keep tiled shapes for input and output nodes
        input_nodes = set(node.all_input_nodes)
        tile_shapes = {
            n: s for n, s in tile_shapes.items()
            if n in input_nodes or n is node
        }

        logger.debug(f"Evaluating banking strategy:")

        for part in self.partitions:
            nodes = [
                key_to_node[key] for key in part.tensors if key in key_to_node
            ]

            if not nodes:
                continue

            logger.debug(f"  Partition tensors: {part.tensors} -> nodes: {nodes}")
            specified_nodes.update(nodes)

            # Make sure the partition starts from a new bank
            if part.align:
                current_offset = _align_size(current_offset, bank_size)

            for n in nodes:
                # For fused operation the output node of the first node becomes
                # an intermediate and will not appear in tile_shapes
                if n not in tile_shapes:
                    continue

                t_size = _get_tensor_size(
                    n, tile_shapes[n], (n is node), bank_width
                )

                node_to_segment[n] = Segment(
                    start=current_offset,
                    end=current_offset + t_size,
                    memory_space=MemorySpace.SCRATCHPAD,
                    node=n
                )

                current_offset += t_size

            if part.align:
                current_offset = _align_size(current_offset, bank_size)

        unspecified_nodes = [n for n in tile_shapes if n not in specified_nodes]
        if not unspecified_nodes:
            return current_offset, node_to_segment

        if self.unspecified_policy == BankingPolicy.SEPARATE_ALIGNED:
            current_offset = _align_size(current_offset, bank_size)

        for n in unspecified_nodes:
            t_size = _get_tensor_size(
                n, tile_shapes[n], (n is node), bank_width
            )

            node_to_segment[n] = Segment(
                start=current_offset,
                end=current_offset + t_size,
                memory_space=MemorySpace.SCRATCHPAD,
                node=n
            )

            current_offset += t_size

            if self.unspecified_policy == BankingPolicy.SEPARATE_ALIGNED:
                current_offset = _align_size(current_offset, bank_size)

        logger.debug(f"  Unspecified tensors: {unspecified_nodes}")
        logger.debug(f"    Unspecified policy: {self.unspecified_policy}")
        logger.debug(f"    Scratchpad segments:")
        for n, seg in node_to_segment.items():
            logger.debug(f"      Node: {n}, Segment: [{seg.start}, {seg.end})")
        logger.debug(f"    Total scratchpad used: {current_offset} bytes")

        return current_offset, node_to_segment


class BankingStrategyRegistry:
    """
    Global registry mapping op_kind -> list of banking strategies.
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, List[BankingStrategy]] = {}

    def register(self, op_kind: Any, strategies: Sequence[BankingStrategy]) -> None:
        self._strategies[op_kind] = list(strategies)

    def get(self, op_kind: str) -> List[BankingStrategy]:
        return self._strategies.get(op_kind, DEFAULT_FALLBACK_STRATEGIES)


BANKING_STRATEGY_REGISTRY = BankingStrategyRegistry()


def register_banking_strategies(targets: Any) -> Callable:
    """
    Decorator for registering banking strategies for one or more op kinds.

    The decorated function must return a list (or tuple) of BankingStrategy.
    """
    target_list = targets if isinstance(targets, (list, tuple)) else [targets]

    def decorator(fn: Callable[[], Sequence[BankingStrategy]]):
        for target in target_list:
            strategies = fn(target)
            if not isinstance(strategies, (list, tuple)):
                raise TypeError(
                    "Banking strategy function must return a list or tuple of BankingStrategy"
                )

            for s in strategies:
                if not isinstance(s, BankingStrategy):
                    raise TypeError("All elements must be BankingStrategy")

            BANKING_STRATEGY_REGISTRY.register(target, strategies)
        return fn

    return decorator


DEFAULT_FALLBACK_STRATEGIES: List[BankingStrategy] = [
    BankingStrategy(
        partitions=(), unspecified_policy=BankingPolicy.SEPARATE_ALIGNED,
    ),
    BankingStrategy(
        partitions=(), unspecified_policy=BankingPolicy.SEPARATE_UNALIGNED,
    ),
]


def _get_scope(target) -> str:
    if hasattr(target, "_schema"):
        scope = target._schema.name.split('::')[1]
    else:
        scope = str(target)
    return scope


@register_banking_strategies([
    torch.ops.aten.conv2d.default,
    torch.ops.aten.linear.default,
    torch.ops.quantized_ops.conv2d.default,
    torch.ops.quantized_ops.linear.default,
])
def _(target) -> List[BankingStrategy]:
    scope = _get_scope(target)
    return [
        BankingStrategy(
            partitions=(), unspecified_policy=BankingPolicy.SEPARATE_ALIGNED,
        ),
        BankingStrategy((
            BankPartition((f"{scope}::weight", f"{scope}::bias")),
        )),
        BankingStrategy((
            BankPartition((
                f"{scope}::weight", f"{scope}::bias", f"{scope}::output"
            )),
        )),
        BankingStrategy(
            partitions=(), unspecified_policy=BankingPolicy.SEPARATE_UNALIGNED,
        ),
    ]


@register_banking_strategies(torch.ops.aten.matmul.default)
def _(target) -> List[BankingStrategy]:
    return [
        BankingStrategy(
            partitions=(), unspecified_policy=BankingPolicy.SEPARATE_ALIGNED,
        ),
        BankingStrategy((
            BankPartition(("matmul::other", "matmul::output")),
        )),
        BankingStrategy(
            partitions=(), unspecified_policy=BankingPolicy.SEPARATE_ALIGNED,
        ),
    ]


@register_banking_strategies(torch.ops.quantized_ops.conv2d_mx.default)
def _(target) -> List[BankingStrategy]:
    return [
        BankingStrategy(
            partitions=(), unspecified_policy=BankingPolicy.SEPARATE_ALIGNED,
        ),
        BankingStrategy((
            BankPartition(("conv2d_mx::weight", "conv2d_mx::weight_scale", "conv2d_mx::bias")),
        )),
        BankingStrategy((
            BankPartition(("conv2d_mx::input", "conv2d_mx::input_scale")),
            BankPartition(("conv2d_mx::weight", "conv2d_mx::weight_scale", "conv2d_mx::bias")),
        )),
        BankingStrategy((
            BankPartition((
                "conv2d_mx::input", "conv2d_mx::input_scale",
                "conv2d_mx::weight", "conv2d_mx::weight_scale", "conv2d_mx::bias",
            )),
        )),
        BankingStrategy(
            partitions=(), unspecified_policy=BankingPolicy.SEPARATE_UNALIGNED,
        ),
    ]


@register_banking_strategies(torch.ops.quantized_ops.linear_mx.default)
def _(target) -> List[BankingStrategy]:
    return [
        BankingStrategy(
            partitions=(), unspecified_policy=BankingPolicy.SEPARATE_ALIGNED,
        ),
        BankingStrategy((
            BankPartition(("linear_mx::weight", "linear_mx::weight_scale", "linear_mx::bias")),
        )),
        BankingStrategy((
            BankPartition(("linear_mx::weight", "linear_mx::weight_scale", "linear_mx::bias")),
            BankPartition(("linear_mx::A_data", "linear_mx::A_indices", "linear_mx::A_indptr")),
        )),
        BankingStrategy((
            BankPartition(("linear_mx::input", "linear_mx::input_scale")),
            BankPartition(("linear_mx::weight", "linear_mx::weight_scale", "linear_mx::bias")),
            BankPartition(("linear_mx::A_data", "linear_mx::A_indices", "linear_mx::A_indptr")),
        )),
        BankingStrategy((
            BankPartition((
                "linear_mx::input", "linear_mx::input_scale",
                "linear_mx::weight", "linear_mx::weight_scale", "linear_mx::bias",
                "linear_mx::A_data", "linear_mx::A_indices", "linear_mx::A_indptr",
            )),
        )),
        BankingStrategy(
            partitions=(), unspecified_policy=BankingPolicy.SEPARATE_UNALIGNED,
        ),
    ]

@register_banking_strategies(torch.ops.quantized_ops.matmul_mx.default)
def _(target) -> List[BankingStrategy]:
    return [
        BankingStrategy(
            partitions=(), unspecified_policy=BankingPolicy.SEPARATE_ALIGNED,
        ),
        BankingStrategy((
            BankPartition(("matmul_mx::other", "matmul_mx::weight_scale")),
        )),
        BankingStrategy((
            BankPartition(("matmul_mx::input", "matmul_mx::input_scale")),
            BankPartition(("matmul_mx::other", "matmul_mx::weight_scale")),
        )),
        BankingStrategy((
            BankPartition((
                "matmul_mx::input", "matmul_mx::input_scale", "matmul_mx::other",
                "matmul_mx::weight_scale",
            )),
        )),
        BankingStrategy(
            partitions=(), unspecified_policy=BankingPolicy.SEPARATE_UNALIGNED,
        ),
    ]


def get_banking_strategies_for_op(op_kind: Any) -> List[BankingStrategy]:
    """
    Retrieve banking strategies for an op kind.
    """
    return BANKING_STRATEGY_REGISTRY.get(op_kind)
