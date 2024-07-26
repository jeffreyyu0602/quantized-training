import operator
from functools import reduce


class Partition:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.node = None  # None means the partition is free

class MemoryManager:
    """
    This class implements a simple first-fit memory manager for allocating memory partitions to tensors.

    Attributes:
        total_memory (int): The total amount of memory available for allocation.
        memory_partitions (list of Partition): A list of memory partitions.
        tensor_memory_map (dict): A dictionary mapping tensors to their allocated memory partitions.

    """
    def __init__(self, total_memory):
        self.total_memory = total_memory
        self.memory_partitions = [Partition(start=0, end=total_memory)]
        self.tensor_memory_map = {}

    def calculate_tensor_size(self, shape):
        if len(shape) == 0:
            return 1
        return reduce(operator.mul, shape)

    def allocate_memory(self, node, size=None):
        if not hasattr(node, 'shape'):
            print(f"Node {node} does not have a shape attribute")
            return None

        # FIXME: assumes double precision
        tensor_size = size or self.calculate_tensor_size(node.shape) * 2
        for partition in self.memory_partitions:
            if partition.node is None and (partition.end - partition.start) >= tensor_size:
                if (partition.end - partition.start) > tensor_size:
                    new_partition = Partition(start=partition.start + tensor_size, end=partition.end)
                    partition.end = partition.start + tensor_size
                    self.memory_partitions.insert(self.memory_partitions.index(partition) + 1, new_partition)
                self.tensor_memory_map[node] = partition
                partition.node = node
                return Partition(start=partition.start, end=partition.end)
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
            status = 'free' if partition.node is None else 'allocated'
            print(f"Partition from {partition.start} to {partition.end} is {status}. Nodes: {partition.node}")
