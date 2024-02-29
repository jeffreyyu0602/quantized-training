import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantized_training.posit import quantize_to_posit


def transpose_for_inputs(x, tile_size):
    padding = (tile_size - x.size(-1) % tile_size) % tile_size
    if padding > 0:
        x = F.pad(x, (0, padding), mode='constant', value=0)
    n = int(x.shape[-1] / tile_size)
    new_x_shape = x.size()[:-1] + (n, tile_size)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)


def transpose_for_weight(x, tile_size):
    padding = (tile_size - x.size(0) % tile_size) % tile_size
    if padding > 0:
        x = F.pad(x, (0, 0, 0, padding), mode='constant', value=0)
    n = int(x.shape[0] / tile_size)
    new_x_shape = (n, tile_size) + x.size()[1:]
    return x.view(new_x_shape)


def tiled_matmul(input, weight, tile_size=16):
    inp_tiled = transpose_for_inputs(input, tile_size)
    w_tiled = transpose_for_weight(weight, tile_size)
    output = torch.empty((*input.shape[:-1], weight.shape[-1]), dtype=input.dtype, device=input.device)
    for i in range(input.shape[0]):
        psum = torch.bmm(inp_tiled[i], w_tiled)
        psum = quantize_to_posit(psum, 16, 1, round_to_even=True)
        output[i] = torch.sum(psum, dim=0)
    return output


def test_matmul():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.randn(3, 18, 147, device=device)
    weight = torch.randn(147, 224, device=device)
    out = tiled_matmul(input, weight, 16)
    gold = torch.matmul(input, weight)
    print((gold - out).abs().max())
    print(torch.allclose(gold, out, atol=1e-4, rtol=1e-4))


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1,
           groups=1):
    print("custom conv2d")
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    padding = (padding, padding) if isinstance(padding, int) else padding
    stride = (stride, stride) if isinstance(stride, int) else stride

    N, _, H_in, W_in = input.shape
    C_out, _, kernel_height, kernel_width = weight.shape

    H_out = math.floor((H_in + 2 * padding[0] - dilation[0] * kernel_height) / stride[0]) + 1
    W_out = math.floor((W_in + 2 * padding[1] - dilation[1] * kernel_width) / stride[1]) + 1

    inp_unf = F.unfold(input, weight.shape[-2:], dilation, padding, stride)
    weight = weight.view(weight.size(0), -1).T
    out_unf = tiled_matmul(inp_unf.transpose(1, 2), weight)
    if bias is not None:
        out_unf += bias
    out = out_unf.transpose(1, 2).view(N, C_out, H_out, W_out)
    return out


def test_conv2d():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.randn(1, 64, 56, 56, device=device)
    weight = torch.randn(64, 64, 3, 3, device=device)
    bias = torch.randn(64, device=device)
    out = conv2d(input, weight, bias=bias, padding=1, fn=torch.matmul)
    gold = F.conv2d(input, weight, bias=bias, padding=1)
    print((gold - out).abs().max())
    print(torch.allclose(gold, out, atol=1e-4, rtol=1e-4))


def test_tiled_conv2d():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.randn(1, 512, 14, 14, device=device).bfloat16()
    weight = torch.randn(512, 512, 3, 3, device=device).bfloat16()
    bias = torch.randn(512, device=device).bfloat16()
    out = conv2d(input, weight, bias=bias, padding=1)
    gold = F.conv2d(input, weight, bias=bias, padding=1)
    print(torch.norm(gold - out))
    print(torch.allclose(gold, out, atol=1e-4, rtol=1e-4))


if __name__ == '__main__':
    torch.manual_seed(42)
    test_matmul()
    # test_conv2d()
    # test_tiled_conv2d()