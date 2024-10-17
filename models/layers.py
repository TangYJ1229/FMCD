from typing import List, Optional

from torch import Tensor, reshape, stack
from .Change_attention import ChangeAttention

from torch.nn import (
    Conv2d,
    InstanceNorm2d,
    Module,
    PReLU,
    Sequential,
    ConvTranspose2d,
    LayerNorm
)


class PixelwiseLinear(Module):
    def __init__(
        self,
        fin: List[int],
        fout: List[int],
        last_activation: Module = None,
    ) -> None:
        assert len(fout) == len(fin)
        super().__init__()

        n = len(fin)
        self._linears = Sequential(
            *[
                Sequential(
                    Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                    PReLU()
                    if i < n - 1 or last_activation is None
                    else last_activation,
                )
                for i in range(n)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # Processing the tensor:
        return self._linears(x)


class MixingBlock(Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
    ):
        super().__init__()
        self.dim = ch_out
        self._convmix = Sequential(
            Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            PReLU(),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Packing the tensors and interleaving the channels:
        mixed = stack((x, y), dim=2)
        mixed = reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))
        mixed = self._convmix(mixed)
        mixed = LayerNorm(mixed.shape[1:]).to('cuda')(mixed)

        # Mixing:
        return mixed


class Up(Module):
    def __init__(
            self,
            nin: int,
            nout: int,
    ):
        super().__init__()
        self._upsample = ConvTranspose2d(nin, nout, 3, 2, 1, output_padding=1)

        self._convolution = Sequential(
            Conv2d(nout, nout, 3, 1, groups=nout, padding=1),
            PReLU(),
            InstanceNorm2d(nout),

            Conv2d(nout, nout, kernel_size=1, stride=1),
            PReLU(),
            InstanceNorm2d(nout),
        )

    def forward(self, x):
        x = self._upsample(x)
        x = self._convolution(x)
        return x


class DomainUp(Module):
    def __init__(
            self,
            nin: int,
            nout: int,
    ):
        super().__init__()
        self._upsample = ConvTranspose2d(nin, nout, 3, 2, 1, output_padding=1)

        self._convolution = Sequential(
            Conv2d(nout, nout, 3, 1, groups=nout, padding=1),
            PReLU(),
            Conv2d(nout, nout, 3, 1, 1),
            PReLU(),
            Conv2d(nout, nout, kernel_size=1, stride=1),
            PReLU(),
        )

    def forward(self, x):
        x = self._upsample(x)
        x = self._convolution(x)
        return x


class MixAttentionBlock(Module):
    def __init__(
            self,
            dim,
    ):
        super().__init__()

        self._changeattn = ChangeAttention(dim=dim)
        self._mixing = MixingBlock(dim*2, dim)
        self._convolution = Sequential(
            Conv2d(dim, dim, 3, 1, groups=dim, padding=1),
            PReLU(),
            InstanceNorm2d(dim),
            Conv2d(dim, dim, kernel_size=1, stride=1),
            PReLU(),
            InstanceNorm2d(dim),
        )

    def forward(self, x, y):
        attn = self._changeattn(x, y)
        mixed = self._mixing(x, y)
        mixed = self._convolution(mixed)

        return mixed, attn


def gram_matrix(feature):
    b, c, h, w = feature.size()
    feature = feature.view(b, c, h*w)
    feature_t = feature.transpose(1, 2)
    gram = feature.bmm(feature_t) / (c*h*w)
    return gram

