import operator
from functools import reduce

import torch


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

    def calculate_tensor_size(self, shape):
        if len(shape) == 0:
            return 1
        return reduce(operator.mul, shape)

    def allocate_memory(self, node, size=None):
        if not hasattr(node, 'shape'):
            print(f"Node {node} does not have a shape attribute")
            return None

        tensor_size = size or self.calculate_tensor_size(node.shape)

        # TODO: this should be removed for proper double precision handling
        # On Minotaur, biases are hardcoded to double precision. Attention masks
        # are implemented as biases. Therefore, they need to be in double
        # precision as well. Softmax inputs use double precision for better accuracy.
        if node.op == "get_attr" and len(node.shape) == 1:
            tensor_size *= 2
            print(f"Node {node} is a bias tensor")

        if node.op == "placeholder" and len([d for d in node.shape if d > 1]) == 1:
            tensor_size *= 2
            print(f"Node {node} is an attention mask")

        if node.op == "placeholder" and len(node.shape) == 4:
            tensor_size *= 2
            print(f"Node {node} requires more space because of replication.")

        if next(iter(node.users)).target == torch.ops.aten.softmax.int:
            tensor_size *= 2
            print(f"Node {node} is an input to softmax")

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
        return None

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
