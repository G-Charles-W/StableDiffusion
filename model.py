from AttnBlock import AttnDownBlock, AttnUpBlock
from ResnetBlock import DownSampleBlock, UpSampleBlock
from MiddleBlock import MiddleBlock
from TimeEmbedding import Timesteps, TimestepEmbedding
from torch import nn
import torch


class Model(nn.Module):

    def __init__(self, in_channels, out_channels,
                 down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
                 up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"), ):
        super().__init__()
        block_out_channels = (224, 448, 672, 896)

        # 1. Calculate time embedding
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(224, 896)

        # 2. pre-process
        self.conv1 = nn.Conv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        # 3. Down-sampling
        res_down = DownSampleBlock(block_out_channels[0], block_out_channels[0])
        attn_down1 = AttnDownBlock(block_out_channels[0], block_out_channels[1], head_dim=8, residual_connect=True)
        attn_down2 = AttnDownBlock(block_out_channels[1], block_out_channels[2], head_dim=8, residual_connect=True)
        attn_down3 = AttnDownBlock(block_out_channels[2], block_out_channels[3], head_dim=8, residual_connect=True,
                                   down_sample=False)
        self.down_blocks = nn.ModuleList([res_down, attn_down1, attn_down2, attn_down3])

        # 4. Middle block
        self.mid_block = MiddleBlock(block_out_channels[3], block_out_channels[3])

        # 5. up
        self.up_blocks = nn.ModuleList([])
        r_out_chs = list(reversed(block_out_channels))
        output_channel = r_out_chs[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = r_out_chs[i]
            input_channel = r_out_chs[min(i + 1, len(block_out_channels) - 1)]

            is_final = (i == len(up_block_types) - 1)
            if up_block_type == 'AttnUpBlock2D':
                self.up_blocks.append(
                    AttnUpBlock(input_channel, output_channel, prev_output_channel, up_sample=not is_final))
            elif up_block_type == "UpBlock2D":
                self.up_blocks.append(
                    UpSampleBlock(input_channel, output_channel, prev_output_channel, up_sample=not is_final))

        # 6.
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-6)

        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, sample, timestep):
        # 1. Calculate time embedding
        time_steps = timestep
        if not torch.is_tensor(time_steps):
            time_steps = torch.tensor([time_steps], dtype=torch.long, device=sample.device)

        time_steps = time_steps * torch.ones(sample.shape[0], dtype=time_steps.dtype, device=time_steps.device)
        t_emb = self.time_proj(time_steps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv1(sample)

        # 3. Down-sampling
        down_block_res_samples = (sample,)
        for i in range(4):
            sample, res_samples = self.down_blocks[i](sample, emb)
            down_block_res_samples += res_samples

        # 4. Middle block
        sample = self.mid_block(sample, emb)

        # 5. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            sample = upsample_block(sample, res_samples, emb)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


if __name__ == '__main__':
    model = Model(3, 1).to("cuda")
    sample_ = torch.randn(1, 3, 64, 64, 64).to("cuda")
    # timesteps = torch.tensor([1], dtype=torch.long).to("cuda")
    with torch.no_grad():
        model(sample_, 1)
