import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from ..pt2e_utils import dtype_byte_size

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    start: int
    end: int
    partition_id: Optional[int] = None
    node: Optional[torch.fx.Node] = None

    def __post_init__(self):
        original_start = self.start
        original_end = self.end

        self.start = int(self.start)
        self.end = math.ceil(self.end)

        if self.start != original_start:
            logger.warning(f"Segment start {original_start} is not an integer. Rounding to {self.start}.")
        if self.end != original_end:
            logger.warning(f"Segment end {original_end} is not an integer. Rounding up to {self.end}.")


def get_user_with_target(node: torch.fx.Node, targets):
    if not isinstance(targets, (list, tuple)):
        targets = [targets]

    for user in node.users:
        if user.target in targets:
            return user

        if user.op == 'call_module':
            gm = user.meta['submodule']
            placeholder = next(n for n in gm.graph.nodes if n.name == node.name)
            user = get_user_with_target(placeholder, targets)
            if user is not None:
                return user
    return None


class MemoryAllocator:
    """
    This class implements a simple first-fit memory manager for allocating memory partitions to tensors.

    Attributes:
        total_memory (int): The total amount of memory available for allocation.
        segments (list of Segment): A list of memory partitions.
        memory_map (dict): A dictionary mapping tensors to their allocated memory partitions.

    """
    total_partitions = 0

    def __init__(self, total_memory=None, bank_width=None, bank_size=None):
        self.total_memory = total_memory or (1 << 63) - 1
        self.bank_width = bank_width
        self.bank_size = bank_size

        self.partition_id = MemoryAllocator.total_partitions
        MemoryAllocator.total_partitions += 1

        self.segments = [Segment(start=0, end=self.total_memory, partition_id=self.partition_id)]
        self.memory_map = {}
        self.snapshots = []

    def align_size(self, size, align_bank=False):
        if self.bank_size is not None and align_bank:
            alignment = self.bank_size
        elif self.bank_width is not None:
            alignment = self.bank_width
        else:
            return size
        return (size + alignment - 1) // alignment * alignment

    def get_tensor_size(self, node, shape, align_bank=False):
        if isinstance(node.value, torch.Tensor):
            dtype = node.meta.get('dtype') or node.value.dtype
            numel = math.prod(shape) if shape is not None else node.value.numel()
            tensor_size = numel * dtype_byte_size(dtype)

            # This logic is going to be removed in the future
            conv2d_user = get_user_with_target(node, [
                torch.ops.aten.conv2d.default,
                torch.ops.quantized_ops.conv2d.default,
            ])

            if conv2d_user is not None:
                input_node = conv2d_user.args[0]
                input_node = input_node.meta.get('source_node', input_node)
                dim = 1 if conv2d_user.target == torch.ops.aten.conv2d.default else -1
                if input_node == node and node.value.shape[dim] < 16:
                    tensor_size *= 2
                    logger.info(
                        f"Conv2d {node} has input with shape: {node.value.shape}. "
                        "Doubling size to perform replication."
                    )

            return self.align_size(tensor_size, align_bank)

        if isinstance(node.value, (tuple, list)):
            dtypes = node.meta.get('dtype', [None] * len(node.value))
            if shape is not None and isinstance(shape, (tuple, list)):
                numel = [math.prod(s) for s in shape]
            else:
                numel = [t.numel() for t in node.value]
            node.meta["output_sizes"] = [
                self.align_size(numel[i] * dtype_byte_size(dtypes[i] or t.dtype))
                for i, t in enumerate(node.value)
            ]
            return self.align_size(sum(node.meta["output_sizes"]), align_bank=align_bank)

        logger.warning(f"Node {node} has a non-tensor output")
        return None

    def allocate_memory(self, node, size=None, shape=None, align_bank=False):
        if not hasattr(node, 'value'):
            print(f"Node {node} does not have a value attribute")
            return None

        # Skip allocation for quantization scaling factors
        quantize_user = get_user_with_target(node, [
            torch.ops.quantized_ops.quantize.default,
            torch.ops.quantized_ops.dequantize.default,
        ])

        if (
            isinstance(node.value, torch.Tensor)
            and node.value.numel() == 1
            and quantize_user is not None
        ):
            logger.info(f"Skipping allocation for scalar scale tensor: {node.name}")
            return None

        tensor_size = (
            size if size is not None else
            self.get_tensor_size(node, shape, align_bank=align_bank)
        )

        if tensor_size is None:
            return None

        for index, segment in enumerate(self.segments):
            if segment.node is None and (segment.end - segment.start) >= tensor_size:
                if (segment.end - segment.start) > tensor_size:
                    new_partition = Segment(
                        start=segment.start + tensor_size,
                        end=segment.end,
                        partition_id=self.partition_id,
                    )
                    segment.end = segment.start + tensor_size
                    self.segments.insert(index + 1, new_partition)
                self.memory_map[node] = segment
                segment.node = node
                return Segment(start=segment.start, end=segment.end, partition_id=self.partition_id)

        raise RuntimeError(f"Memory allocation failed for tensor {node.name}")

    def free_memory(self, node):
        if node in self.memory_map:
            segment = self.memory_map[node]
            segment.node = None
            self.merge_segments()
            del self.memory_map[node]

    def merge_segments(self):
        i = 0
        while i < len(self.segments) - 1:
            current_partition = self.segments[i]
            next_partition = self.segments[i + 1]
            if current_partition.node is None and next_partition.node is None:
                current_partition.end = next_partition.end
                self.segments.pop(i + 1)
            else:
                i += 1

    def print_layout(self):
        for segment in self.segments:
            status = 'free' if segment.node is None else segment.node.name
            print(f"Segment from {segment.start} to {segment.end}: {status}")

    def snapshot(self):
        partitions = [
            Segment(start=segment.start, end=segment.end, node=segment.node.name if segment.node else None)
            for segment in self.segments[:-1]
        ]
        self.snapshots.append(partitions)

    def dump_snapshots(self, filename="dump_snapshot.png", colormap_name='tab20'):
        """
        Plots memory usage over time from a list of memory snapshots.
        Tensors (partitions with nodes) cycle through colors from the colormap.
        Free partitions are shown in gray.
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        cmap = cm.get_cmap(colormap_name)
        color_cycle = [cmap(i) for i in range(cmap.N)]
        free_color = (0.85, 0.85, 0.85)

        id_to_color = {}
        color_index = 0

        for t, snapshot in enumerate(self.snapshots):
            for segment in snapshot:
                if segment.node is None:
                    color = free_color
                else:
                    if segment.node not in id_to_color:
                        id_to_color[segment.node] = color_cycle[color_index % len(color_cycle)]
                        color_index += 1
                    color = id_to_color[segment.node]

                ax.bar(
                    x=t,
                    height=segment.end - segment.start,
                    width=1.0,
                    bottom=segment.start,
                    color=color,
                    linewidth=0,
                )

        def format_bytes(x, _):
            if x >= 1 << 30:
                return f"{x / (1 << 30):.0f}GB"
            elif x >= 1 << 20:
                return f"{x / (1 << 20):.0f}MB"
            elif x >= 1 << 10:
                return f"{x / (1 << 10):.0f}KB"
            else:
                return f"{x:.0f}B"

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_bytes))

        max_bytes = max(p.end for snapshot in self.snapshots for p in snapshot)
        max_mb = max_bytes / (1 << 20)

        # Auto interval using base-10 logic
        locator = mticker.MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True)
        tick_vals_mb = locator.tick_values(0, max_mb)
        tick_vals_bytes = [int(mb * (1 << 20)) for mb in tick_vals_mb]

        ax.set_yticks(tick_vals_bytes)

        ax.set_title("Active Memory Timeline")
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
