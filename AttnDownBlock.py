import torch
from torch import nn
import torch.nn.functional as F
from ResnetBlock import ResnetBlock


class AttnDownBlock(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 groups: int = 32,
                 heads=84,
                 temb_channels: int = 896,
                 eps: float = 1e-6, num_layers: int = 2):
        super().__init__()
        resnets = []
        attentions = []
        in_ch_s = [in_channels, out_channels]
        out_ch_s = [out_channels, out_channels]
        for i in range(num_layers):
            resnets.append(
                ResnetBlock(
                    in_channels=in_ch_s[i],
                    out_channels=out_ch_s[i],
                    temb_channels=temb_channels,
                    eps=eps,
                    groups=groups
                )
            )
            attentions.append(
                Attention(
                    out_channels,
                    heads=heads,
                    eps=eps,
                    bias=True,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, hidden_states, temb):
        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states = output_states + (hidden_states,)


class Attention(nn.Module):

    def __init__(self, channels,
                 groups: int = 32,
                 heads=84,
                 bias=True,
                 eps: float = 1e-6):
        super().__init__()
        self.group_norm = torch.nn.GroupNorm(num_groups=groups, num_channels=channels, eps=eps, affine=True)
        self.to_q = nn.Linear(channels, channels, bias=bias)
        self.to_k = nn.Linear(channels, channels, bias=bias)
        self.to_v = nn.Linear(channels, channels, bias=bias)
        self.heads = heads
        self.to_out = nn.Linear(channels, channels, bias=True)

    def forward(self, hidden_states, encoder_hidden_states=None):
        residual = hidden_states
        batch_size, channel, height, width, length = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width*length).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = self.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = self.to_out(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width, length)
        return hidden_states
