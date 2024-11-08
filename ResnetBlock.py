import torch
from torch import nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 groups: int = 32,
                 temb_channels: int = 896,
                 eps: float = 1e-6):
        super().__init__()

        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.non_linearity = nn.SiLU()
        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)

        self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden_states = x

        # Group Norm + Silu + Conv
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.non_linearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        # integrate time embedding
        temb = self.time_emb_proj(temb)[:, :, None, None, None]
        hidden_states = hidden_states + temb

        # Group Norm + Silu + Conv
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.non_linearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        # Resnet Shortcut
        x = self.conv_shortcut(x)
        output_tensor = (x + hidden_states)
        return output_tensor


class DownSampleBlock(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 groups=32,
                 temb_channels: int = 896,
                 eps: float = 1e-6):
        super().__init__()

        resnet1 = ResnetBlock(in_channels, out_channels, groups, temb_channels, eps)
        resnet2 = ResnetBlock(out_channels, out_channels, groups, temb_channels, eps)
        self.down_sample = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.resnets = nn.ModuleList([resnet1, resnet2])

    def forward(self, x, temb):
        output_states = ()
        hidden_states = x
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)
        hidden_states = self.down_sample(hidden_states)
        output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 prev_output_channel,
                 groups=32,
                 temb_channels: int = 896,
                 eps: float = 1e-6,
                 up_sample=False,
                 num_layers: int = 2):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            # res_skip
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnets.append(
                ResnetBlock(in_channels=resnet_in_channels + res_skip_channels,
                            out_channels=out_channels, temb_channels=temb_channels, eps=eps,
                            groups=groups))
            # resnets.append(
            #     ResnetBlock(in_channels=in_channels[i] + res_skip_channels[i],
            #                 out_channels=out_channels[i], temb_channels=temb_channels, eps=eps,
            #                 groups=groups))
        # resnet1 = ResnetBlock(in_channels, out_channels, groups, temb_channels, eps)
        # resnet2 = ResnetBlock(out_channels, out_channels, groups, temb_channels, eps)
        # self.down_sample = nn.Conv3d(out_channels[-1], out_channels[-1], kernel_size=3, stride=2, padding=1, bias=True)
        self.resnets = nn.ModuleList(resnets)
        self.up_sample_flag = up_sample
        if self.up_sample_flag:
            self.up_sample = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=True)

    def forward(self, x, res_hidden_states_tuple, temb):
        hidden_states = x
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.up_sample_flag:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            hidden_states = self.up_sample(hidden_states)

        return hidden_states


if __name__ == '__main__':
    # block_out_channels = (224, 448, 672, 896)
    # resnet = ResnetBlock(224, 224, 32)
    #
    # sample = torch.randn(1, 224, 128, 128, 128)
    # # timesteps = torch.tensor([1], dtype=torch.long)
    # # timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)
    # #
    # # time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0)
    # # time_embedding = TimestepEmbedding(224, 896)
    # #
    # # t_emb = time_proj(timesteps)
    # # emb = time_embedding(t_emb)
    dhs_chs = [224, 224, 224]
    dhs_size = [128, 128, 128]
    downsample_hidden_states = ()
    emb = torch.randn(1, 896)
    for i in range(len(dhs_chs)):
        # ch = int(dhs_size[i]/2)
        ch = dhs_size[i]
        hidden_state = torch.randn(1, dhs_chs[i], ch, ch, ch).cuda()
        downsample_hidden_states = downsample_hidden_states + (hidden_state,)
    # print(resnet(sample, emb).shape)
    x = torch.randn(1, 448, 128, 128, 128).to("cuda")
    emb = torch.randn(1, 896).to("cuda")
    in_ch = [448, 224, 224]
    out_ch = [224, 224, 224]
    res_skip = [224, 224, 224]
    # up_sampler = UpSampleBlock(in_channels=in_ch, out_channels=out_ch, res_skip_channels=res_skip).cuda()
    up_sampler = UpSampleBlock(in_channels=224, out_channels=224, prev_output_channel=448).cuda()
    with torch.no_grad():
        up_sampler(x, downsample_hidden_states, emb)
