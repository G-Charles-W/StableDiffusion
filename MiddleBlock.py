import torch
from torch import nn
from ResnetBlock import ResnetBlock
from AttnBlock import Attention


class MiddleBlock(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 groups=32,
                 temb_channels: int = 896,
                 eps: float = 1e-6,
                 head_dim: int = 8):
        super().__init__()
        heads = out_channels // head_dim
        self.resnet1 = ResnetBlock(in_channels, out_channels, groups, temb_channels, eps)
        self.resnet2 = ResnetBlock(out_channels, out_channels, groups, temb_channels, eps)
        self.attn = Attention(out_channels, groups, heads)

    def forward(self, x, temb):
        hidden_states = self.resnet1(x, temb)
        hidden_states = self.attn(hidden_states)
        hidden_states = self.resnet2(hidden_states, temb)

        return hidden_states
