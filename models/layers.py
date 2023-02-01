import torch
from torch import nn, Tensor


class DoubleConv(nn.Module):
  
    def __init__(self, inp_ch: int, out_ch: int) -> None:
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
        nn.Conv2d(inp_ch, out_ch, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace = True),

        nn.Conv2d(out_ch, out_ch, 3, padding = 1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace = True))

    def forward(self, x) -> Tensor:
        return self.conv(x)


class InputConvolution(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InputConvolution, self).__init__()
        self.inp_conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.inp_conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size = 2, stride = 2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        x = torch.cat([x2, x1], dim = 1)
        x = self.conv(x)

        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class LastConvolution(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LastConvolution, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv1(x)
        return x

