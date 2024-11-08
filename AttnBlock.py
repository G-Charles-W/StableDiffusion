import torch
from torch import nn
import torch.nn.functional as F
from ResnetBlock import ResnetBlock


class AttnDownBlock(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 groups: int = 32,
                 head_dim=8,
                 temb_channels: int = 896,
                 eps: float = 1e-6,
                 num_layers: int = 2,
                 residual_connect=False,
                 down_sample=True):
        super().__init__()

        resnets = []
        attentions = []
        in_ch_s = [in_channels, out_channels]
        out_ch_s = [out_channels, out_channels]
        heads = out_channels // head_dim
        self.down_sample_flag = down_sample

        for i in range(num_layers):
            resnets.append(
                ResnetBlock(in_channels=in_ch_s[i], out_channels=out_ch_s[i], temb_channels=temb_channels, eps=eps,
                            groups=groups))

            attentions.append(
                Attention(out_channels, heads=heads, eps=eps, bias=True, residual_connect=residual_connect))

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if self.down_sample_flag:
            self.down_sample = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, hidden_states, temb):
        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states = output_states + (hidden_states,)

        if self.down_sample_flag:
            hidden_states = self.down_sample(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class AttnUpBlock(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 prev_output_channel,
                 groups: int = 32,
                 head_dim=8,
                 temb_channels: int = 896,
                 eps: float = 1e-6,
                 num_layers: int = 3,
                 residual_connect=False,
                 up_sample=True):
        super().__init__()

        resnets = []
        attentions = []
        heads = out_channels // head_dim
        self.up_sample_flag = up_sample

        for i in range(num_layers):
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels

            resnets.append(
                ResnetBlock(in_channels=resnet_in_channels + res_skip_channels,
                            out_channels=out_channels, temb_channels=temb_channels, eps=eps,
                            groups=groups))

            attentions.append(
                Attention(out_channels, heads=heads, eps=eps, bias=True, residual_connect=residual_connect))

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if self.up_sample_flag:
            self.up_sample = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=True)

    def forward(self, hidden_states, res_hidden_states_tuple, temb):
        # output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            # output_states = output_states + (hidden_states,)

        if self.up_sample_flag:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            hidden_states = self.up_sample(hidden_states)
            # output_states = output_states + (hidden_states,)

        return hidden_states


class Attention(nn.Module):

    def __init__(self, channels,
                 groups: int = 32,
                 heads=84,
                 bias=True,
                 eps: float = 1e-6,
                 residual_connect=False):
        super().__init__()
        self.group_norm = torch.nn.GroupNorm(num_groups=groups, num_channels=channels, eps=eps, affine=True)
        self.to_q = nn.Linear(channels, channels, bias=bias)
        self.to_k = nn.Linear(channels, channels, bias=bias)
        self.to_v = nn.Linear(channels, channels, bias=bias)
        self.heads = heads
        self.to_out = nn.Linear(channels, channels, bias=True)
        self.residual_connect = residual_connect

    def forward(self, hidden_states, encoder_hidden_states=None):
        residual = hidden_states
        batch_size, channel, height, width, length = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width * length).transpose(1, 2)

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

        if self.residual_connect:
            hidden_states += residual
        return hidden_states


if __name__ == '__main__':
    dhs_chs = [224, 224, 224, 224, 448, 448, 448, 672, 672, 672, 896, 896]
    dhs_size = [128, 128, 128, 64, 64, 64, 32, 32, 32, 16, 16, 16]
    downsample_hidden_states = ()
    attn1 = AttnUpBlock(672, 896, 896)
    attn2 = AttnUpBlock(672, 896, 896)
    attn3 = AttnUpBlock(672, 896, 896)
    up1 = AttnUpBlock(672, 896, 896)

    model = []
    print(model)
    for i in range(len(dhs_chs)):
        ch = int(dhs_size[i]/2)
        hidden_state = torch.randn(1, dhs_chs[i], ch, ch, ch).cuda()
        downsample_hidden_states = downsample_hidden_states + (hidden_state,)
    x = torch.randn(1, 896, 8, 8, 8).to("cuda")
    emb = torch.randn(1, 896).to("cuda")

    print(model(x, downsample_hidden_states, emb).shape)
