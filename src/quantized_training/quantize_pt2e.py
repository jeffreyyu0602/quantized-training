import copy
from dataclasses import asdict, replace
from typing import Optional

import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.fx.utils import assert_and_get_unique_device

from typing import Dict
from torch.ao.quantization.quantizer import (
    EdgeOrNode,
    SharedQuantizationSpec,
    QuantizationSpecBase,
)
from torch.ao.quantization import ObserverOrFakeQuantize

import quantized_training as qt
from quantized_training.export_utils import _allow_exported_model_train_eval
from quantized_training.fake_quantize import FusedAmaxObsFakeQuantize
from quantized_training.quantizer.quantizer import QuantizationSpec
from quantized_training.quantizer.xnnpack_quantizer import XNNPACKQuantizer
from quantized_training.quantizer.xnnpack_quantizer_utils import QuantizationConfig


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


def _get_microscaling_quantizer(
    activation: Optional[QuantizationSpec],
    weight: Optional[QuantizationSpec],
):
    # Microscaling performs quantization along the reduction dimension
    act_qspec = replace(activation, ch_axis=1)
    weight_qspec = replace(weight, ch_axis=1)
    conv2d_qonfig = QuantizationConfig(act_qspec, None, weight_qspec, None)

    act_qspec = replace(activation, ch_axis=-1)
    weight_qspec = replace(weight, ch_axis=-1)
    linear_qonfig = QuantizationConfig(act_qspec, None, weight_qspec, None)

    act0_qspec = replace(activation, ch_axis=-1)
    act1_qspec = replace(activation, ch_axis=-2)
    matmul_qonfig = QuantizationConfig(act0_qspec, None, act1_qspec, None)

    quantizer = (XNNPACKQuantizer()
                 .set_module_type(torch.nn.Conv2d, conv2d_qonfig)
                 .set_module_type(torch.nn.Linear, linear_qonfig)
                 .set_operator_type(torch.ops.aten.matmul.default, matmul_qonfig))
    return quantizer


def _get_per_channel_act_quantizer(
    activation: Optional[QuantizationSpec],
    weight: Optional[QuantizationSpec],
):
    # Convolution layer only support per-tensor activation quantization
    act_qspec = replace(activation, qscheme=qt.per_tensor_symmetric)
    conv2d_qonfig = QuantizationConfig(act_qspec, None, weight, None)

    # Perform quantization along the outer dimension
    act_qspec = replace(activation, ch_axis=-2)
    linear_qonfig = QuantizationConfig(act_qspec, None, weight, None)

    act0_qspec = replace(activation, ch_axis=-2)
    act1_qspec = replace(activation, ch_axis=-1)
    matmul_qonfig = QuantizationConfig(act0_qspec, None, act1_qspec, None)

    quantizer = (XNNPACKQuantizer()
                 .set_module_type(torch.nn.Conv2d, conv2d_qonfig)
                 .set_module_type(torch.nn.Linear, linear_qonfig)
                 .set_operator_type(torch.ops.aten.matmul.default, matmul_qonfig))
    return quantizer


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

    if qt.per_vector_symmetric in qschemes or qt.microscaling in qschemes:
        assert len(set(qschemes)) == 1, (
            f"Quantization scheme {qschemes[0]} does not work with {qschemes[1]}"
        )
        return _get_microscaling_quantizer(activation, weight)

    if weight is not None and weight.qscheme == qt.per_channel_symmetric:
        assert weight.ch_axis == 0, (
            f"Per-channel weight quantization only supports quantizing output "
            "channel dimension (dim=0)."
        )

    if activation is not None and activation.qscheme == qt.per_channel_symmetric:
        return _get_per_channel_act_quantizer(activation, weight)

    qconfig_opt = QuantizationConfig(activation, None, weight, None)
    matmul_qonfig = QuantizationConfig(activation, None, activation, None)
    return XNNPACKQuantizer().set_global(qconfig_opt) \
        .set_operator_type(torch.ops.aten.matmul.default, matmul_qonfig)


def prepare_pt2e(model, quantizer, args, kwargs=None, dynamic_shapes=None):
    from torch.ao.quantization.pt2e import prepare
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e

    # HACK monkey patching to replace the default implementation of _create_obs_or_fq_from_qspec
    prepare._get_obs_or_fq_map = _get_obs_or_fq_map

    # Make sure captured model is on the same device as the original model
    model_device = assert_and_get_unique_device(model)
    model = capture_pre_autograd_graph(
        model,
        args=args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
    ).to(model_device)

    model = prepare_pt2e(model, quantizer)
    _allow_exported_model_train_eval(model)
    return model
