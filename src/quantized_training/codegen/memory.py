import logging
from random import randint

import torch
import matplotlib.pyplot as plt

from ..pt2e_utils import dtype_byte_size

logger = logging.getLogger(__name__)


class Partition:
    def __init__(self, start, end, partition_id=None):
        self.start = start
        self.end = end
        self.partition_id = partition_id
        self.node = None  # None means the partition is free

    def __str__(self):
        return f"Partition(start={self.start}, end={self.end}, partition_id={self.partition_id})"


class MemoryManager:
    """
    This class implements a simple first-fit memory manager for allocating memory partitions to tensors.

    Attributes:
        total_memory (int): The total amount of memory available for allocation.
        memory_partitions (list of Partition): A list of memory partitions.
        tensor_memory_map (dict): A dictionary mapping tensors to their allocated memory partitions.

    """
    total_partitions = 0

    def __init__(self, total_memory):
        self.total_memory = total_memory
        self.partition_id = MemoryManager.total_partitions
        MemoryManager.total_partitions += 1
        self.memory_partitions = [Partition(start=0, end=total_memory, partition_id=self.partition_id)]
        self.tensor_memory_map = {}
        self.snapshots = []  # Stores the snapshots of partitions over time

    def allocate_memory(self, node, size=None):
        if not hasattr(node, 'value'):
            print(f"Node {node} does not have a value attribute")
            return None

        def is_conv2d_input(n):
            if n.op == 'get_attr':
                return False
            for user in n.users:
                if user.target == torch.ops.aten.conv2d.default:
                    return True
                elif user.op == 'call_module':
                    submodule = user.meta['submodule']
                    for sub_node in submodule.graph.nodes:
                        if sub_node.name == n.name and is_conv2d_input(sub_node):
                            return True
            return False

        tensor_size = size
        if tensor_size is None:
            if isinstance(node.value, torch.Tensor):
                tensor_size = size or node.value.numel()
                if node.meta.get('dtype', None) is not None:
                    tensor_size *= dtype_byte_size(node.meta['dtype'])
                else:
                    tensor_size *= dtype_byte_size(node.value.dtype)
            elif isinstance(node.value, (tuple, list)):
                tensor_size = sum(t.numel() * dtype_byte_size(t.dtype) for t in node.value)
            else:
                logger.warning(f"Node {node} has a non-tensor output")
                return None

            # TODO the systolic array dimension is hardcoded here
            if is_conv2d_input(node) and node.value.shape[1] < 16:
                print(f"Node {node} requires replication. Increase memory size.")
                tensor_size *= 2

        # torch.bool has 1/8 byte. Round total size to the nearest byte.
        tensor_size = int(tensor_size)

        for partition in self.memory_partitions:
            if partition.node is None and (partition.end - partition.start) >= tensor_size:
                if (partition.end - partition.start) > tensor_size:
                    new_partition = Partition(
                        start=partition.start + tensor_size,
                        end=partition.end,
                        partition_id=self.partition_id,
                    )
                    partition.end = partition.start + tensor_size
                    self.memory_partitions.insert(self.memory_partitions.index(partition) + 1, new_partition)
                self.tensor_memory_map[node] = partition
                partition.node = node
                return Partition(start=partition.start, end=partition.end, partition_id=self.partition_id)

        raise RuntimeError(f"Memory allocation failed for tensor {node.name}")

    def free_memory(self, node):
        partition = self.tensor_memory_map[node]
        partition.node = None
        self.merge_partitions()
        del self.tensor_memory_map[node]

    def merge_partitions(self):
        i = 0
        while i < len(self.memory_partitions) - 1:
            current_partition = self.memory_partitions[i]
            next_partition = self.memory_partitions[i + 1]
            if current_partition.node is None and next_partition.node is None:
                current_partition.end = next_partition.end
                self.memory_partitions.pop(i + 1)
            else:
                i += 1

    def print_partitions(self):
        for partition in self.memory_partitions:
            status = 'free' if partition.node is None else partition.node.name
            print(f"Partition from {partition.start} to {partition.end}: {status}")

    def take_snapshot(self):
        partitions = [
            (partition.start, partition.end, partition.node.name if partition.node else 'Free')
            for partition in self.memory_partitions
        ]
        self.snapshots.append(partitions)


def visualize_memory_layout(snapshots, filename="memory_layout.png"):
    color_map = plt.cm.get_cmap('tab10')
    node_colors = {}

    def get_color(node_name):
        if node_name not in node_colors:
            color_idx = randint(0, color_map.N)
            node_colors[node_name] = color_map(color_idx)
        return node_colors[node_name]

    num_snapshots = len(snapshots)
    peak_memory = max(p[1] for snapshot in snapshots for p in snapshot if p[2] != 'Free')

    plt.rcParams.update({'font.size': 20})

    fig, ax = plt.subplots(figsize=(10, num_snapshots * 0.5))
    ax.set_xlim(0, peak_memory)
    ax.set_ylim(-0.5, num_snapshots - 0.5)
    ax.set_yticks(range(num_snapshots))

    for i, snapshot in enumerate(snapshots):
        for (start, end, label) in snapshot:
            color = get_color(label) if label != 'Free' else (0, 0, 0, 0)
            ax.barh(y=i, width=(end - start), left=start, color=color, height=1)
            if i < num_snapshots - 1:
                ax.hlines(y=i + 0.5, xmin=0, xmax=peak_memory, color='black', linewidth=1)

    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
