from torch import nn
import torch
from torchvision.models import vgg16
from torch.nn.functional import l1_loss
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

def MS_SSIMLoss(out_image, gt_image):
    return 1 - ms_ssim(out_image, gt_image, data_range=1, size_average=True)

class L_TV(nn.Module):
    def __init__(self):
        super(L_TV, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

class IlluminationSmoothnessLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, enhanced_image):
        grad_x = torch.abs(enhanced_image[:, :, :-1] - enhanced_image[:, :, 1:])
        grad_y = torch.abs(enhanced_image[:, :-1, :] - enhanced_image[:, 1:, :])
        loss = torch.mean(grad_x * grad_x) + torch.mean(grad_y * grad_y)
        return loss

class ColorConsistencyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, original_image, enhanced_image):
        mean_orig = torch.mean(original_image, dim=(2, 3), keepdim=True)
        mean_enhanced = torch.mean(enhanced_image, dim=(2, 3), keepdim=True)


        loss = F.mse_loss(mean_orig, mean_enhanced)
        return loss




def angle(a, b):
    vector = torch.mul(a, b)
    up     = torch.sum(vector)
    down   = torch.sqrt(torch.sum(torch.square(a))) * torch.sqrt(torch.sum(torch.square(b)))
    theta  = torch.acos(up/down)
    return theta

def color_loss(out_image, gt_image):
    loss = torch.mean(angle(out_image[:,0,:,:],gt_image[:,0,:,:]) +
                      angle(out_image[:,1,:,:],gt_image[:,1,:,:]) +
                      angle(out_image[:,2,:,:],gt_image[:,2,:,:]))
    return loss


class L_exp(nn.Module):
    def __init__(self, patch_size=16, mean_val=0.6):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        loss = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return loss

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vgg = vgg16(pretrained=True).features
        vgg = vgg.to(device)
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:16])
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, enhanced_image, target_image):
        enhanced_features = self.vgg_layers(enhanced_image)
        target_features = self.vgg_layers(target_image)
        loss = l1_loss(enhanced_features, target_features)
        return loss

def exposure_loss(img1, img2):
    luminance1 = 0.2126 * img1[:, 0, :, :] + 0.7152 * img1[:, 1, :, :] + 0.0722 * img1[:, 2, :, :]
    luminance2 = 0.2126 * img2[:, 0, :, :] + 0.7152 * img2[:, 1, :, :] + 0.0722 * img2[:, 2, :, :]

    loss = F.mse_loss(luminance1, luminance2)
    return loss



def Total_loss(out_image, gt_image):
    mae_loss = F.l1_loss(out_image,gt_image)
    ms_ssim_loss = MS_SSIMLoss(out_image, gt_image)
    smooth_loss = L_TV()
    smooth_loss_val = smooth_loss(out_image)
    color_loss_val = color_loss(out_image, gt_image)
    perceptual_loss = PerceptualLoss()
    perceptual_loss_val = perceptual_loss(out_image, gt_image)

    exp_loss = exposure_loss(out_image, gt_image)

    Ill= IlluminationSmoothnessLoss()
    ill_loss =  Ill(out_image)

    color_consistent  = ColorConsistencyLoss()
    color_consistent_loss = color_consistent(out_image, gt_image)

    weights = [0, 1,0,0,0]
    weighted_loss = (weights[0] * mae_loss+
                     weights[1] * ms_ssim_loss +
                     weights[2] * smooth_loss_val +
                     weights[3] * color_loss_val +
                     weights[4] * perceptual_loss_val)
    return weighted_loss

































