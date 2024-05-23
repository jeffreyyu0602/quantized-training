import torch
from torch._export import capture_pre_autograd_graph

from .fake_quantize import FusedAmaxObsFakeQuantize
from .quantizer import QuantizationSpec
from .quantizer.xnnpack_quantizer import QuantizationConfig, XNNPACKQuantizer


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
    kwargs = quantization_spec.to_dict()
    kwargs.pop("observer_or_fake_quant_ctr")
    return FusedAmaxObsFakeQuantize.with_args(**kwargs)()


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

    qconfig = QuantizationConfig(args.activation, args.error, args.weight, None)
    quantizer = XNNPACKQuantizer().set_global(qconfig)

    prepare_pt2e(model, quantizer)
    model.recompile()
    # model.graph.print_tabular()
    return model
