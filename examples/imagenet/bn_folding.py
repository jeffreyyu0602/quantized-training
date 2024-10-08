import copy

import torch
import torchvision.models as models

# This file provides bn_folding_model, a function to fold bn into conv
# With minor changes taken from here https://github.com/raja-kumar/folding-batchnorm/blob/master/bn_folder.py


def bn_folding_model(model):

    new_model = copy.deepcopy(model)

    module_names = list(new_model._modules)

    for k, name in enumerate(module_names):

        if len(list(new_model._modules[name]._modules)) > 0:
            new_model._modules[name] = bn_folding_model(
                new_model._modules[name])

        else:
            if isinstance(new_model._modules[name], torch.nn.BatchNorm2d):
                if isinstance(new_model._modules[module_names[k-1]], torch.nn.Conv2d):

                    # Folded BN
                    folded_conv = fold_conv_bn_eval(
                        new_model._modules[module_names[k-1]], new_model._modules[name])

                    # Remove the BN layer
                    new_model._modules[(name)] = torch.nn.Identity()
                    new_model._modules[module_names[k-1]] = folded_conv

    return new_model


def bn_folding(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = bn_rm.new_zeros(bn_rm.shape)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    w_fold = conv_w * (bn_w * bn_var_rsqrt).view(-1, 1, 1, 1)
    b_fold = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(w_fold), torch.nn.Parameter(b_fold)


def fold_conv_bn_eval(conv, bn):
    assert (not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = bn_folding(fused_conv.weight, fused_conv.bias,
                                                    bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv
