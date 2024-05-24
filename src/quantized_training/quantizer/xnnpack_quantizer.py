from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer import Quantizer
from torch.fx import Node

from .xnnpack_quantizer_utils import (
    _convert_scalars_to_attrs,
    OP_TO_ANNOTATOR,
    QuantizationConfig,
)


__all__ = [
    "XNNPACKQuantizer",
]


def _get_module_name_filter(module_name: str):
    """Get the module_name_filter function for a given module name, the filter accepts
    a node and checks if the node comes from a module that has certain module name

    For example:
        node: linear_op = call_function[...](...)  # comes from a module with name blocks.sub.linear1


    >> module_name_filter = _get_module_name_filter("blocks.sub")
    >> print(module_name_filter(node))
    True  # the node is from "blocks.sub" based on the fully qualified name "blocks.sub.linear1"
    """

    def module_name_filter(n: Node) -> bool:
        # example: {
        #    'L__self___sub': ("L['self'].sub", <class '....Sub'>),
        #    'L__self___sub_linear': ("L['self'].sub.linear", <class 'torch.nn.modules.linear.Linear'>)
        # }
        # get_attr nodes doesn't have nn_module_stack?
        nn_module_stack = n.meta.get("nn_module_stack", {})

        def _normalize_path(n):
            prefix = 0
            # TODO This is non standard behavior and should be removed when we migrate off capture_pre_autograd_graph.
            if n.startswith("L['self']."):
                prefix = len("L['self'].")
            return n[prefix:]

        names = [_normalize_path(n) for n, _ in nn_module_stack.values()]
        return module_name in names

    return module_name_filter


def _get_module_type_filter(tp: Callable):
    """Get the module_type_filter function for a given module type, the filter accepts
    a node and checks if the node comes from a module that has certain module type

    For example:
        node: linear_op = call_function[...](...)  # comes from a module with type Block -> Sub -> Linear


    >> module_type_filter = _get_module_type_filter(Sub)  # submodule with type `Sub`, under the `Block` submodule
    >> print(module_type_filter(node))
    True  # the node is from the submodule `Sub` (same for `Block` and `Linear` as well)
    """

    tp_str = tp.__module__ + "." + tp.__qualname__

    def module_type_filter(n: Node) -> bool:
        # example: {
        #     'L__self___sub': ("L['self'].sub", <class '....Sub'>),
        #     'L__self___sub_linear': ("L['self'].sub.linear", <class 'torch.nn.modules.linear.Linear'>)
        # }
        nn_module_stack = n.meta.get("nn_module_stack", {})
        types = []
        for _, t in nn_module_stack.values():
            # export() returns str, but older APIs (e.g. capture_pre_autograd_graph)
            # return type. Handle both cases.
            if isinstance(t, type):
                t = t.__module__ + "." + t.__qualname__
            types.append(t)
        return tp_str in types

    return module_type_filter


def _get_operator_type_filter(op: torch._ops.OpOverloadPacket):
    """Get the operator_type_filter function for a given operator type, the filter accepts
    a node and checks if the node comes from a call_function node that calls the operator

    For example:
        node: linear_op = call_function[...](...)  # calls the linear operator


    >> operator_type_filter = _get_operator_type_filter(torch.ops.aten.linear.default)
    >> print(operator_type_filter(node))
    True  # the node calls the linear operator
    """

    def operator_type_filter(n: Node) -> bool:
        return n.target == op

    return operator_type_filter


def _get_not_module_type_or_name_filter(
    tp_list: List[Callable], module_name_list: List[str]
) -> Callable[[Node], bool]:
    module_type_filters = [_get_module_type_filter(tp) for tp in tp_list]
    module_name_list_filters = [_get_module_name_filter(m) for m in module_name_list]

    def not_module_type_or_name_filter(n: Node) -> bool:
        return not any(f(n) for f in module_type_filters + module_name_list_filters)

    return not_module_type_or_name_filter


class XNNPACKQuantizer(Quantizer):
    # static quantization ops (both PTQ and QAT)
    # Preserve the order that fusions come before singular ops
    STATIC_OPS = [
        "linear",
        "conv",
        "matmul",
    ]

    def __init__(self):
        super().__init__()
        self.global_config: Optional[QuantizationConfig] = None
        self.operator_type_config: Dict[
            torch._ops.OpOverloadPacket, Optional[QuantizationConfig]
        ] = {}
        self.module_type_config: Dict[Callable, Optional[QuantizationConfig]] = {}
        self.module_name_config: Dict[str, Optional[QuantizationConfig]] = {}

    def set_global(self, quantization_config: QuantizationConfig) -> XNNPACKQuantizer:
        self.global_config = quantization_config
        return self

    def set_operator_type(
        self,
        operator_type: torch._ops.OpOverloadPacket,
        quantization_config: QuantizationConfig,
    ) -> XNNPACKQuantizer:
        self.operator_type_config[operator_type] = quantization_config
        return self

    def set_module_type(
        self, module_type: Callable, quantization_config: QuantizationConfig
    ):
        """Set quantization_config for a submodule with type: `module_type`, for example:
        quantizer.set_module_name(Sub) or quantizer.set_module_name(nn.Linear), it will quantize all supported operator/operator
        patterns in the submodule with this module type with the given `quantization_config`
        """
        self.module_type_config[module_type] = quantization_config
        return self

    def set_module_name(
        self, module_name: str, quantization_config: Optional[QuantizationConfig]
    ):
        """Set quantization_config for a submodule with name: `module_name`, for example:
        quantizer.set_module_name("blocks.sub"), it will quantize all supported operator/operator
        patterns in the submodule with this module name with the given `quantization_config`
        """
        assert (
            quantization_config is not None
        ), " quantization_config == None is not supported yet"
        self.module_name_config[module_name] = quantization_config
        return self

    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        """Transforms scalar values to tensor attributes"""
        return _convert_scalars_to_attrs(model)

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        module_name_list = list(self.module_name_config.keys())
        for module_name, config in self.module_name_config.items():
            self._annotate_all_static_patterns(
                model, config, _get_module_name_filter(module_name)
            )

        tp_list = list(self.module_type_config.keys())
        for module_type, config in self.module_type_config.items():
            self._annotate_all_static_patterns(
                model, config, _get_module_type_filter(module_type)
            )

        for op, config in self.operator_type_config.items():
            self._annotate_all_static_patterns(
                model, config, _get_operator_type_filter(op)
            )

        self._annotate_all_static_patterns(
            model,
            self.global_config,
            _get_not_module_type_or_name_filter(tp_list, module_name_list),
        )
        return model

    def _annotate_all_static_patterns(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[Callable[[Node], bool]] = None,
    ) -> torch.fx.GraphModule:
        # TODO: implement the support for None to be canceling out previous annotations
        if quantization_config is None:
            return model

        for op in self.STATIC_OPS:
            OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass
