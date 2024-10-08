import re
from collections import defaultdict
from typing import Dict, Tuple, Callable, Any

import torch
from torch._export import capture_pre_autograd_graph
from torch.fx import GraphModule, Node
from accelerate.big_modeling import infer_auto_device_map
from accelerate.utils import get_max_memory


__all__ = [
    "dispatch_model",
    "dtype_byte_size",
    "get_device_map",
]


def dtype_byte_size(dtype: torch.dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.

    Example:

    ```py
    >>> dtype_byte_size(torch.float32)
    4
    ```
    """
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)(_.*)?$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def get_device_map(model: GraphModule, max_memory=None, verbose=False):
    if max_memory is None:
        max_memory = get_max_memory(max_memory)

    devices = list(max_memory.keys())
    if "disk" not in devices:
        devices.append("disk")

    device_map = {}
    current_device = 0
    current_memory_used = 0

    nodes_to_treat = [n for n in model.graph.nodes if n.op == 'get_attr']
    while len(nodes_to_treat) > 0:
        node = nodes_to_treat.pop(0)
        if verbose:
            print(f"\nTreating node {node}.")
        # Assess size needed
        tensor = node.meta.get('val')
        module_size = tensor.numel() * dtype_byte_size(tensor.dtype)

        device = devices[current_device]
        current_max_size = max_memory[device] if device != "disk" else None

        if current_max_size is not None and current_memory_used + module_size > current_max_size:
            if verbose:
                print(
                    f"Not enough space on {devices[current_device]} to put {node} (space available "
                    f"{current_max_size-current_memory_used}, module size {module_size})."
                )
            current_device += 1
            assert current_device < len(devices), "Not enough devices to store the model."
            nodes_to_treat = [node] + nodes_to_treat
            current_memory_used = 0
        else:
            if verbose:
                if current_max_size is None:
                    print(f"Putting {node} (size={module_size}) on {devices[current_device]}.")
                else:
                    print(
                        f"Putting {node} (size={module_size}) on {devices[current_device]} "
                        f"(available={current_max_size-current_memory_used})."
                    )
            current_memory_used += module_size
            device_map[node.target] = devices[current_device]

    return device_map


def dispatch_model(model, args, device_map=None, max_memory=None):
    if device_map is None:
        device_map = infer_auto_device_map(model, max_memory=max_memory)

    args_iter = iter(args)
    env : Dict[str, Node] = {}
    modules = dict(model.named_modules())

    def load_arg(a):
        return torch.fx.graph.map_arg(a, lambda n: env[n])

    def fetch_attr(node):
        if 'param_constant' in node.name:
            return model.get_parameter(node.target)
        return model.get_buffer(node.target)

    def is_node_visited(node):
        if isinstance(node, (list, tuple)):
            return all(is_node_visited(n) for n in node)
        return node in env

    def get_devices(tensor):
        devices = []
        if isinstance(tensor, torch.Tensor):
            devices.append(tensor.device)
        elif isinstance(tensor, (list, tuple)):
            for t in tensor:
                devices.extend(get_devices(t))
        return devices

    def insert_adaptor(node, user, env, device):
        if isinstance(node, (list, tuple)):
            for n in node:
                insert_adaptor(n, user, env, device)
            return

        if not isinstance(node, Node):
            return

        value = env[node]
        if not isinstance(value, torch.Tensor) or value.device == device:
            return

        to_node = None
        for n in node.users:
            if n.target == torch.Tensor.to and n.args[1] == device:
                to_node = n

        if to_node is None:
            with model.graph.inserting_after(node):
                to_node = model.graph.call_function(
                    torch.Tensor.to, (node, device))
            env[to_node] = value.to(device)

        user.replace_input_with(node, to_node)

    for node in model.graph.nodes:
        if node.op == 'placeholder':
            result = next(args_iter)
        elif node.op == 'get_attr':
            device = device_map[node.target]
            attribute = fetch_attr(node)
            attribute.data = attribute.data.to(device)
            result = attribute.data
        elif node.op == 'call_function':
            args = load_arg(node.args)
            devices = set(get_devices(args))
            if len(devices) > 1:
                # Sort the devices and use the last one
                devices = sorted(str(d) for d in devices)
                device = torch.device(devices[-1])
                for arg in node.args:
                    insert_adaptor(arg, node, env, device)
            result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
        elif node.op == 'call_module':
            result = modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

        env[node] = result

        # Free up memory
        for n in list(env.keys()):
            if n.target != torch.Tensor.to and is_node_visited(list(n.users.keys())):
                env[n] = None

    model.graph.lint()
    model.recompile()


def get_aten_graph_module(
    pattern: Callable,
    example_inputs: Tuple[Any, ...],
    is_cuda: bool = False,
    **kwargs,
) -> GraphModule:
    """
    Convert the pattern to an FX graph with decomposed aten ops.
    """
    if is_cuda:
        example_inputs = tuple([x.cuda() if isinstance(x, torch.Tensor) else x for x in example_inputs])
    aten_pattern = capture_pre_autograd_graph(
        pattern,
        example_inputs,
        kwargs,
    )
    aten_pattern.graph.eliminate_dead_code()
    aten_pattern.recompile()
    return aten_pattern


def get_node_name_to_scope(model: GraphModule) -> Dict[str, Tuple[str, type, int]]:
    node_name_to_scope: Dict[str, Tuple[str, type]] = {}
    submodule_to_object_type_to_cur_idx: Dict[str, Dict[Callable, int]] = \
        defaultdict(lambda: defaultdict(int))
    for n in model.graph.nodes:
        if (nn_module_stack := n.meta.get("nn_module_stack", None)) is None:
            node_name_to_scope[n.name] = [("", type(None))]
            continue

        def _normalize_path(n):
            prefix = 0
            # TODO This is non standard behavior and should be removed when we migrate off capture_pre_autograd_graph.
            if n.startswith("L['self']."):
                prefix = len("L['self'].")
            return n[prefix:]

        current_scope = []
        for bt in nn_module_stack.values():
            module_path = _normalize_path(bt[0])
            cur_object_type_idx = \
                submodule_to_object_type_to_cur_idx[module_path][n.target]
            submodule_to_object_type_to_cur_idx[module_path][n.target] += 1
            current_scope.append((module_path, bt[1], cur_object_type_idx))
        node_name_to_scope[n.name] = current_scope[-1]

    return node_name_to_scope


def print_node_scope_tabular(gm: GraphModule):
    try:
        from tabulate import tabulate
    except ImportError:
        print("`print_tabular` relies on the library `tabulate`, "
                "which could not be found on this machine. Run `pip "
                "install tabulate` to install the library.")
        raise

    node_name_to_scope = get_node_name_to_scope(gm)
    node_specs = [
        [n.op, n.name, n.target, node_name_to_scope[n.name]]
        for n in gm.graph.nodes if n.name in node_name_to_scope
    ]
    print(tabulate(node_specs,
                   headers=['opcode', 'name', 'target', 'scope']))
