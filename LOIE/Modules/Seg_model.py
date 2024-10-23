
import torch
import torch.nn as nn
import torch.nn.functional as F


class LISU_DECOMP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv_5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)
        self.conv_6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_8 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        c1 = self.conv_1(img)
        c2 = F.relu(self.conv_2(c1))
        c3 = F.relu(self.conv_3(c2))
        c4 = F.relu(self.conv_4(c3))
        c5 = F.relu(self.conv_5(c4))

        # Adjusting size of c5 to match c2 for concatenation
        if c5.size()[2:] != c2.size()[2:]:
            c5 = F.interpolate(c5, size=c2.size()[2:], mode='bilinear', align_corners=False)

        c6 = F.relu(self.conv_6(torch.cat((c2, c5), dim=1)))

        # Adjusting size of c6 to match c1 for concatenation
        if c6.size()[2:] != c1.size()[2:]:
            c6 = F.interpolate(c6, size=c1.size()[2:], mode='bilinear', align_corners=False)

        c7 = self.conv_7(torch.cat((c1, c6), dim=1))
        c8 = self.conv_8(c7)
        c9 = torch.sigmoid(c8)

        reflectance, illumination = c9[:, :3, :, :], c9[:, 3:4, :, :]

        return illumination, reflectance


class LISU_JOINT(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 4

        # Encoder layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # decoder
        self.mid_r = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(), )

        self.mid_s = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(), )

        self.deconv4_r = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.deconv3_r = nn.Sequential(
            nn.Conv2d(768, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )


        self.deconv2_r = nn.Sequential(
            nn.Conv2d(384, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.deconv1_r = nn.Sequential(
            nn.Conv2d(192, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'), )

        self.deconv4_s = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.deconv3_s = nn.Sequential(
            nn.Conv2d(768, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.deconv2_s = nn.Sequential(
            nn.Conv2d(384, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.deconv1_s = nn.Sequential(
            nn.Conv2d(192, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'), )

        self.out_r = nn.Sequential(
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
        )

        self.out_s = nn.Sequential(
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        xmid_r = torch.cat((self.mid_r(x5), x5), dim=1)
        xmid_s = torch.cat((self.mid_s(x5), x5), dim=1)

        x4d_r = self.deconv4_r(xmid_r)
        x4d_s = self.deconv4_s(xmid_s)

        # Adjust size for x4 before concatenation
        if x4d_r.size()[2:] != x4.size()[2:]:
            x4d_r = F.interpolate(x4d_r, size=x4.size()[2:], mode='bilinear', align_corners=False)
        if x4d_s.size()[2:] != x4.size()[2:]:
            x4d_s = F.interpolate(x4d_s, size=x4.size()[2:], mode='bilinear', align_corners=False)

        x3d_r = self.deconv3_r(torch.cat([x4d_r, x4d_s, x4], dim=1))
        x3d_s = self.deconv3_s(torch.cat([x4d_s, x4d_r, x4], dim=1))

        # Adjust size for x3 before concatenation
        if x3d_r.size()[2:] != x3.size()[2:]:
            x3d_r = F.interpolate(x3d_r, size=x3.size()[2:], mode='bilinear', align_corners=False)
        if x3d_s.size()[2:] != x3.size()[2:]:
            x3d_s = F.interpolate(x3d_s, size=x3.size()[2:], mode='bilinear', align_corners=False)

        x2d_r = self.deconv2_r(torch.cat([x3d_r, x3d_s, x3], dim=1))
        x2d_s = self.deconv2_s(torch.cat([x3d_s, x3d_r, x3], dim=1))

        # Adjust size for x2 before concatenation
        if x2d_r.size()[2:] != x2.size()[2:]:
            x2d_r = F.interpolate(x2d_r, size=x2.size()[2:], mode='bilinear', align_corners=False)
        if x2d_s.size()[2:] != x2.size()[2:]:
            x2d_s = F.interpolate(x2d_s, size=x2.size()[2:], mode='bilinear', align_corners=False)

        x1d_r = self.deconv1_r(torch.cat([x2d_r, x2d_s, x2], dim=1))
        x1d_s = self.deconv1_s(torch.cat([x2d_s, x2d_r, x2], dim=1))

        # Adjust size for x1 before concatenation
        if x1d_r.size()[2:] != x1.size()[2:]:
            x1d_r = F.interpolate(x1d_r, size=x1.size()[2:], mode='bilinear', align_corners=False)
        if x1d_s.size()[2:] != x1.size()[2:]:
            x1d_s = F.interpolate(x1d_s, size=x1.size()[2:], mode='bilinear', align_corners=False)

        out_r = torch.sigmoid(self.out_r(torch.cat([x1d_r, x1d_s, x1], dim=1)))
        out_s = self.out_s(torch.cat([x1d_s, x1d_r, x1], dim=1))

        return out_s