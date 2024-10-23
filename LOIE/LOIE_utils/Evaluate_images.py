import os
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from LOIE_utils.NIQE import calculate_niqe

# 计算 PSNR
def calculate_psnr(img1, img2):
    return psnr(img1, img2, data_range=img2.max() - img2.min())

# 计算 SSIM
def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True)

# 计算 MAE
def calculate_mae(img1, img2):
    return np.mean(np.abs(img1 - img2))

# 读取文件夹中的图片并计算所有指标
# 读取文件夹中的图片并计算所有指标
def evaluate_images(result_folder, gt_folder):
    result_images = sorted(os.listdir(result_folder))
    gt_images = sorted(os.listdir(gt_folder))

    metrics = {'niqe': [], 'psnr': [], 'ssim': [], 'mae': []}

    for result_img, gt_img in zip(result_images, gt_images):
        result_path = os.path.join(result_folder, result_img)
        gt_path = os.path.join(gt_folder, gt_img)

        # 读取图像
        result_image = Image.open(result_path).convert('RGB')
        gt_image = Image.open(gt_path).convert('RGB')

        # 将图像转换为 numpy 数组
        result_image_np = np.array(result_image).astype(np.float32) / 255.0
        gt_image_np = np.array(gt_image).astype(np.float32) / 255.0

        # 计算 NIQE
        niqe_score = calculate_niqe(result_image_np, crop_border=0, params_path='../niqemodels/')
        # 计算 PSNR
        psnr_score = calculate_psnr(result_image_np, gt_image_np)
        # 计算 SSIM
        ssim_score = calculate_ssim(result_image_np, gt_image_np)
        # 计算 MAE
        mae_score = calculate_mae(result_image_np, gt_image_np)

        # 记录结果
        metrics['niqe'].append(niqe_score)
        metrics['psnr'].append(psnr_score)
        metrics['ssim'].append(ssim_score)
        metrics['mae'].append(mae_score)

        print(
            f"Processed {result_img}: NIQE={niqe_score:.4f}, PSNR={psnr_score:.4f}, SSIM={ssim_score:.4f}, MAE={mae_score:.4f}")

    return metrics


# 使用函数
result_folder = '../train_images/train_image_lunwen/H_result'
gt_folder = '../data/train/train_H'
metrics = evaluate_images(result_folder, gt_folder)

print("Final Metrics:")
print(f"Average NIQE: {np.mean(metrics['niqe']):.4f}")
print(f"Average PSNR: {np.mean(metrics['psnr']):.4f}")
print(f"Average SSIM: {np.mean(metrics['ssim']):.4f}")
print(f"Average MAE: {np.mean(metrics['mae']):.4f}")
