('AttnUpBlock2D', 'AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D')
import torch
from torch import nn
from ResnetBlock import ResnetBlock


class UpSampleBlock(nn.Module):

    def __init__(self):
        super().__init__()
