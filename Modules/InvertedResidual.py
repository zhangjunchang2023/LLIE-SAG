import torch.nn as nn

class DownSample(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        x = self.downsample(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=2):
        super().__init__()
        hidden_dim = round(inp * expand_ratio)
        self.identity = inp == oup

        self.conv = nn.Sequential(
            # pointwise
             nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
             nn.BatchNorm2d(hidden_dim),
             nn.ReLU6(inplace=True),

             # depthwise
             nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
             nn.BatchNorm2d(hidden_dim),
             nn.ReLU6(inplace=True),

             # pointwise linear
             nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
             nn.BatchNorm2d(oup),
            )
        self.downsample = DownSample(inp,oup)

    def forward(self, x):
        x_old = x
        if (self.identity):
            return x_old + self.conv(x)
        else:
            return self.downsample(x_old) + self.conv(x)
