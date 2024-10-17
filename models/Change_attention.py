import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_

def conv_shape(x):
    B, N, C = x.shape
    H = int(math.sqrt(N))
    W = int(math.sqrt(N))
    x = x.reshape(B, C, H, W).contiguous()
    return x


class ChangeAttention(nn.Module):
    def __init__(self, dim, qkv_bias=False):
        super().__init__()
        self.dim = dim

        self.x1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.x2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        x = self.x1(x.flatten(2).transpose(1, 2))
        y = self.x2(y.flatten(2).transpose(1, 2))
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        attn = F.cosine_similarity(x, y, dim=-1)
        attn = 1 - self.sigmoid(attn)
        attn = conv_shape(attn.unsqueeze(dim=-1))

        return attn