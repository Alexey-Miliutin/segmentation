import torch
from torch import nn
from models.layers import *


class UNet(nn.Module):
    def __init__(self, inp_ch: int, out_ch: int, filters: int = 32):
        super(UNet, self).__init__()

        self.inp = InputConvolution(inp_ch, filters)
        self.down1 = Down(filters, filters * 2)
        self.down2 = Down(filters * 2, filters * 4)
        self.down3 = Down(filters * 4, filters * 8)
        self.down4 = Down(filters * 8, filters * 16)
        self.up1 = Up(filters * 16, filters * 8)
        self.up2 = Up(filters * 8, filters * 4)
        self.up3 = Up(filters * 4, filters * 2)
        self.up4 = Up(filters * 2, filters)
        self.out = LastConvolution(filters, out_ch)


    def forward(self, x) -> torch.Tensor:
        x1 = self.inp(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x_out = self.out(x9)
        return x_out