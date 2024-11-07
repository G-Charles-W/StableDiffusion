from torch import nn
# import diffusers
from diffusers import UNet2DModel
import torch


class SelfAttn(nn.Module):

    def __init__(self):
        super().__init__()
        self.to_qkv = nn.Linear(in_features=448, out_features=448, bias=self.use_bias)
        pass

    def forward(self, x):
        return self.to_qkv(x)
        pass


if __name__ == '__main__':
    # model = UNet2DConditionModel(sample_size=(64, 64), in_channels=4, out_channels=1)
    model = UNet2DModel(sample_size=(64, 64), in_channels=4, out_channels=1)
    # model = SelfAttn()
    print(model)
    #
    # print(model(torch.Tensor(size=(57466, 448))))
    timesteps = torch.LongTensor([10])
    # timesteps = torch.randn(1, 320)
    print(model(torch.randn(1, 4, 64, 64), timestep =timesteps))