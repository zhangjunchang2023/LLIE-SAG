import torch
from torch import nn
import torch.nn.functional as F
from Modules.CBAM import CBAM
from Modules.InvertedResidual import InvertedResidual
from Modules.Conv_blocks import DoubleConvDS, DownDS, UpDS, OutConv

class SpatialEncoding(nn.Module):
    def __init__(self, channels):
        super(SpatialEncoding, self).__init__()
        self.channels = channels

    def forward(self, x):
        b, c, h, w = x.size()
        y_grid, x_grid = torch.meshgrid(torch.linspace(0, 1, h, device=x.device),
                                        torch.linspace(0, 1, w, device=x.device))
        y_grid = y_grid.unsqueeze(0).expand(b, 1, h, w)
        x_grid = x_grid.unsqueeze(0).expand(b, 1, h, w)
        pos_encoding = torch.cat([x_grid, y_grid], dim=1)
        pos_encoding = F.interpolate(pos_encoding, size=(h, w), mode='bilinear', align_corners=False)
        return pos_encoding

# Enhance_Core_Net
class Enhance_Core_Net(nn.Module):
    def __init__(self, reduction_ratio=16, scale_factor=1, pos_encoding_channels=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.spatial_encoding = SpatialEncoding(pos_encoding_channels)
        self.inc = DoubleConvDS(3+ pos_encoding_channels, 64)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        self.down4 = DownDS(512, 512)
        self.cbam5 = CBAM(512, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 256)
        self.cbam6 = CBAM(256, reduction_ratio=reduction_ratio)
        self.up2 = UpDS(512, 128)
        self.cbam7 = CBAM(128, reduction_ratio=reduction_ratio)
        self.up3 = UpDS(256, 64)
        self.cbam8 = CBAM(64, reduction_ratio=reduction_ratio)
        self.up4 = UpDS(128, 64)
        self.cbam9 = CBAM(64, reduction_ratio=reduction_ratio)
        self.outc = OutConv(64, 3)

    def forward(self, x, Scale, Shift):
        L_image =x
        pos_encoding = self.spatial_encoding(x)
        x1 = torch.cat([x, pos_encoding], dim=1)
        x1 = self.inc(x1)
        x1 = torch.add(torch.mul(x1, Scale + 0.1), Shift)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        xup1 = self.up1(x5Att, x4Att)
        xup1 = self.cbam6(xup1)
        xup2 = self.up2(xup1, x3Att)
        xup2 = self.cbam7(xup2)
        xup3 = self.up3(xup2, x2Att)
        xup3 = self.cbam8(xup3)
        xup4 = self.up4(xup3, x1Att)
        xup4 = self.cbam9(xup4)
        xup4 = torch.add(torch.mul(xup4, Scale + 0.1), Shift)
        x_r = F.leaky_relu(self.outc(xup4))
        enhance_image = self.enhance(L_image, x_r)
        return enhance_image

    def enhance(self, x, x_r):
        enhanced_x = x + x_r * (torch.pow(x, 2) - x)
        enhanced_x = torch.clamp(enhanced_x, min=0, max=1)
        return enhanced_x
class LOIE_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.S_F0 = InvertedResidual(1,32)
        self.Scale = InvertedResidual(32,64)
        self.S_F2 = InvertedResidual(1,32)
        self.Shift = InvertedResidual(32,64)
        self.enhance_core_net = Enhance_Core_Net()

    def forward(self,L_image,input_S):
        S_F0 = F.leaky_relu(self.S_F0(input_S))
        Scale = F.leaky_relu(self.Scale(S_F0))
        S_F2 = F.leaky_relu(self.S_F2(input_S))
        Shift = torch.sigmoid(self.Shift(S_F2))
        out_image =  self.enhance_core_net(L_image,Scale, Shift)
        return out_image

if __name__ == "__main__":
   model = LOIE_Net()
   print(model)
