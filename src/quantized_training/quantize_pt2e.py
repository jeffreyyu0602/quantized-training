import copy
import logging
from dataclasses import asdict, replace

import torch
from torch._export import capture_pre_autograd_graph

from .fake_quantize import FusedAmaxObsFakeQuantize
from .quantizer import QScheme, QuantizationSpec
from .quantizer.xnnpack_quantizer import QuantizationConfig, XNNPACKQuantizer


logger = logging.getLogger(__name__)


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


def prepare_pt2e(
        model, args, example_args, example_kwargs=None, dynamic_shapes=None):
    # TODO: monkey patching to replace the default implementation of _create_obs_or_fq_from_qspec
    import torch.ao.quantization.fx.prepare
    import sys
    sys.modules["torch.ao.quantization.fx.prepare"]._create_obs_or_fq_from_qspec = \
        _create_obs_or_fq_from_qspec

    from torch.ao.quantization.quantize_pt2e import prepare_pt2e

    model = capture_pre_autograd_graph(
        model,
        args=example_args,
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )

    observer_or_fake_quant_ctr = FusedAmaxObsFakeQuantize.with_args(
        record_histogram=args.record_histogram,
        force_scale_power_of_two=args.force_scale_power_of_two,
    )

    qschemes = []
    if args.activation is not None:
        qschemes.append(args.activation.qscheme)
        args.activation.observer_or_fake_quant_ctr = observer_or_fake_quant_ctr
    if args.weight is not None:
        qschemes.append(args.weight.qscheme)
        args.weight.observer_or_fake_quant_ctr = observer_or_fake_quant_ctr

    if QScheme.PER_VECTOR_SYMMETRIC in qschemes or QScheme.MICROSCALING in qschemes:
        assert len(set(qschemes)) == 1, (
            f"Quantization scheme {qschemes[0]} does not work with {qschemes[1]}"
        )

        # Microscaling performs quantization along the reduction dimension
        act_qspec = replace(args.activation, ch_axis=1) if args.activation else None
        weight_qspec = replace(args.weight, ch_axis=1) if args.weight else None
        conv2d_qonfig = QuantizationConfig(act_qspec, None, weight_qspec, None)

        act_qspec = replace(args.activation, ch_axis=-1) if args.activation else None
        weight_qspec = replace(args.weight, ch_axis=-1) if args.weight else None
        linear_qonfig = QuantizationConfig(act_qspec, None, weight_qspec, None)

        act0_qspec = replace(args.activation, ch_axis=-1) if args.activation else None
        act1_qspec = replace(args.activation, ch_axis=-2) if args.activation else None
        matmul_qonfig = QuantizationConfig(act0_qspec, None, act1_qspec, None)

        quantizer = (XNNPACKQuantizer()
                     .set_module_type(torch.nn.Conv2d, conv2d_qonfig)
                     .set_module_type(torch.nn.Linear, linear_qonfig)
                     .set_operator_type(torch.ops.aten.matmul.default, matmul_qonfig))
    else:
        if (
            args.activation is not None
            and args.activation.qscheme == QScheme.PER_CHANNEL_SYMMETRIC
            and args.activation.ch_axis is not None
        ):
            logger.warning(
                "Per-tensor activation quantization does not support specifying channel axis. "
                "Input ch_axis will be ignored."
            )

        if args.weight is not None and args.weight.qscheme == QScheme.PER_CHANNEL_SYMMETRIC:
            assert args.weight.ch_axis == 0, (
                f"Per-channel weight quantization only supports quantizing output channel dimension (dim=0)."
            )

        act_qspec = replace(args.activation, qscheme=QScheme.PER_TENSOR_SYMMETRIC) if args.activation else None
        conv2d_qonfig = QuantizationConfig(act_qspec, None, args.weight, None)

        act_qspec = replace(args.activation, ch_axis=-2) if args.activation else None
        linear_qonfig = QuantizationConfig(act_qspec, None, args.weight, None)

        act0_qspec = replace(args.activation, ch_axis=-2) if args.activation else None
        act1_qspec = replace(args.activation, ch_axis=-1) if args.activation else None
        matmul_qonfig = QuantizationConfig(act0_qspec, None, act1_qspec, None)

        quantizer = (XNNPACKQuantizer()
                     .set_module_type(torch.nn.Conv2d, conv2d_qonfig)
                     .set_module_type(torch.nn.Linear, linear_qonfig)
                     .set_operator_type(torch.ops.aten.matmul.default, matmul_qonfig))

    prepare_pt2e(model, quantizer)
    model.recompile()
    return model
