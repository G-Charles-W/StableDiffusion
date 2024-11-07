from AttnDownBlock import AttnDownBlock
from DownSampleBlock import DownSampleBlock
from TimeEmbedding import Timesteps, TimestepEmbedding
from torch import nn
import torch


class Model(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        block_out_channels = (224, 448, 672, 896)
        self.conv1 = nn.Conv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)
        down_block1 = DownSampleBlock(block_out_channels[0], block_out_channels[0])
        attn_down1 = AttnDownBlock(block_out_channels[0], block_out_channels[1], heads=56, residual_connect=True)
        attn_down2 = AttnDownBlock(block_out_channels[1], block_out_channels[2], heads=84, residual_connect=True)
        attn_down3 = AttnDownBlock(block_out_channels[2], block_out_channels[3], heads=112, residual_connect=True)

        self.down_blocks = nn.ModuleList(
            [down_block1, attn_down1, attn_down2, attn_down3]
        )
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(224, 896)

    def forward(self, x, timestep):

        # 1. Calculate time embedding
        time_steps = timestep
        if not torch.is_tensor(time_steps):
            time_steps = torch.tensor([time_steps], dtype=torch.long, device=x.device)

        time_steps = time_steps * torch.ones(x.shape[0], dtype=time_steps.dtype, device=time_steps.device)
        t_emb = self.time_proj(time_steps)
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embedding(t_emb)

        # 2. Down-sampling
        out = ()
        hidden_states = self.conv1(x)
        for i in range(4):
            hidden_states = self.down_blocks[i](hidden_states, emb)
            out += (hidden_states,)
        return out


if __name__ == '__main__': 
    model = Model(3).to("cuda")
    sample = torch.randn(1, 3, 128, 128, 128).to("cuda")
    # timesteps = torch.tensor([1], dtype=torch.long).to("cuda")
    model(sample, 1)
