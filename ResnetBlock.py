import torch
from torch import nn


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


if __name__ == '__main__':
    block_out_channels = (224, 448, 672, 896)
    resnet = ResnetBlock(224, 224, 32)

    sample = torch.randn(1, 224, 128, 128, 128)
    # timesteps = torch.tensor([1], dtype=torch.long)
    # timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)
    #
    # time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0)
    # time_embedding = TimestepEmbedding(224, 896)
    #
    # t_emb = time_proj(timesteps)
    # emb = time_embedding(t_emb)
    emb = torch.randn(1, 896)
    print(resnet(sample, emb).shape)
