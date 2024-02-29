import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantized_training import get_default_qconfig


def _transpose_for_inputs(x, tile_size):
    x = x.reshape(-1, x.shape[-1])
    padding = (tile_size - x.shape[-1] % tile_size) % tile_size
    if padding > 0:
        x = F.pad(x, (0, padding), mode='constant', value=0)
    n = int(x.shape[-1] / tile_size)
    new_x_shape = (x.shape[0], n, tile_size)
    x = x.view(new_x_shape)
    return x.permute(1, 0, 2)

def _transpose_for_weight(x, tile_size):
    padding = (tile_size - x.size(0) % tile_size) % tile_size
    if padding > 0:
        x = F.pad(x, (0, 0, 0, padding), mode='constant', value=0)
    n = int(x.shape[0] / tile_size)
    new_x_shape = (n, tile_size) + x.size()[1:]
    return x.view(new_x_shape)

class TiledLinear(nn.Module):
    def __init__(self, tile_size=16, qconfig=None, ):
        super(TiledLinear, self).__init__()
        self.tile_size = tile_size
        self.qconfig = qconfig
        self.fake_quant = qconfig.activation() if qconfig is not None else nn.Identity()

    def matmul(self, input, weight):
        inp_tiled = _transpose_for_inputs(input, self.tile_size)
        w_tiled = _transpose_for_weight(weight.T, self.tile_size)
        output = torch.bmm(inp_tiled, w_tiled)
        output = self.fake_quant(output)
        output = torch.sum(output, dim=0)
        output_shape = input.shape[:-1] + weight.T.shape[1:]
        return output.reshape(output_shape)

    def forward(self, input, weight, bias=None):
        output = self.matmul(input, weight)
        if bias is not None:
            output += bias
        return output

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.randn(128, 512, device=device)
    weight = torch.randn(512, 128, device=device)
    qconfig = get_default_qconfig("posit16_1", activation=True)

    linear = TiledLinear(qconfig=qconfig).to(device)

    out = linear.forward(input, weight)
    gold = torch.matmul(input, weight)

    torch.set_printoptions(precision=10)
    print(torch.sqrt(torch.sum(torch.square(gold - out))))
    print(torch.allclose(gold, out, atol=1e-4, rtol=1e-4))
    print(gold)
    print(out)

def minotaur_matmul():
    pass

if __name__ == '__main__':
    main()