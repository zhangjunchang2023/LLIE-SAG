# PSNR calculation
import math

import torch
import torch.nn.functional as F


#psnr
def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / torch.sqrt(mse))

# MAE calculation
def calculate_mae(img1, img2):
    return F.l1_loss(img1, img2).item()