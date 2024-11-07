import torch
from torch import nn
from ResnetBlock import ResnetBlock


class DownSampleBlock(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 groups=32,
                 temb_channels: int = 896,
                 eps: float = 1e-6):
        super().__init__()

        self.resnet1 = ResnetBlock(in_channels, out_channels, groups, temb_channels, eps)
        self.resnet2 = ResnetBlock(out_channels, out_channels, groups, temb_channels, eps)
        self.down_sample = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x, temb):
        hidden_states = self.resnet1(x, temb)
        hidden_states = self.resnet2(hidden_states, temb)
        hidden_states = self.down_sample(hidden_states)

        return hidden_states


if __name__ == '__main__':
    model = DownSampleBlock(224, 224)
    sample = torch.randn(1, 224, 128, 128, 128)
    emb = torch.randn(1, 896)
    print(model(sample, emb).shape)
