import hashlib
from typing import OrderedDict

import torch
import torch.nn as nn
from torch.fx import GraphModule

from quantized_training.quantize_pt2e import prepare_pt2e


def hash_value(value):
    if isinstance(value, dict):
        return hash_dict(value)
    elif isinstance(value, list):
        return hash_list(value)
    elif isinstance(value, torch.Tensor):
        return str(tuple(value.shape))
    else:
        return str(value)


def hash_list(lst):
    hash_md5 = hashlib.md5()
    for item in lst:
        item = hash_value(item)
        hash_md5.update(item.encode())
    return hash_md5.hexdigest()


def hash_dict(d):
    items = sorted(d.items())
    hash_md5 = hashlib.md5()
    for key, value in items:
        value = hash_value(value)
        hash_md5.update(str(key).encode())
        hash_md5.update(value.encode())
    return hash_md5.hexdigest()


def _hash_args_kwargs(args, kwargs):
    args_hash = hash_list(args)  # Hash the list of positional arguments
    kwargs_hash = hash_dict(kwargs)  # Hash the dict of keyword arguments
    combined_hash = hashlib.md5()
    combined_hash.update(args_hash.encode())
    combined_hash.update(kwargs_hash.encode())
    return combined_hash.hexdigest()


def _get_device(args, kwargs):
    args_devices = [arg.device for arg in args if hasattr(arg, "device")]
    kwargs_devices = [karg.device for karg in kwargs.values() if hasattr(karg, "device")]
    devices = set(args_devices + kwargs_devices)
    assert len(devices) <= 1, "All inputs must be on the same device"
    device = next(iter(devices)) if len(devices) > 0 else None
    return device


class QuantizedModel(nn.Module):

    def __init__(self, model, quantizer, args, kwargs=None, dynamic_shapes=None):
        super().__init__()
        self.quantizer = quantizer
        self.dynamic_shapes = dynamic_shapes

        # Store the original model using a dict so that its modules and parameters
        # are not counted as attributes of this class
        self._orig_model: OrderedDict[GraphModule, None] = {}
        self._orig_model.setdefault(model)
        self._exported_models: OrderedDict[str, GraphModule] = {}
        self.model: GraphModule = None

        self._register_model(args, kwargs)

    def forward(self, *args, **kwargs):
        try:
            output = self.model(*args, **kwargs)
        except TypeError:
            print("Arguments changed. Switching model")
            model = self._register_model(args, kwargs)
            output = model(*args, **kwargs)
        return output

    def _register_model(self, args, kwargs):
        model_id = _hash_args_kwargs(args, kwargs)
        device = _get_device(args, kwargs) or "cpu"
        if (new_model := self._exported_models.get(model_id)) is None:
            model = next(iter(self._orig_model)).to(device)
            new_model = prepare_pt2e(
                model, self.quantizer, args, kwargs, self.dynamic_shapes
            )
            torch.ao.quantization.allow_exported_model_train_eval(new_model)
            self._exported_models[model_id] = new_model

        if self.model is not None:
            state_dict = self.model.state_dict()
            # TODO: some of the state_dict keys may not be present in the new model
            new_model.load_state_dict(state_dict)
            new_model.to(device)
        self.model = new_model
        return new_model
