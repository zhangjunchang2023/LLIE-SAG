import os
import matplotlib
from LOIE_utils.EarlyStopping import EarlyStopping
from LOIE_utils.Init_weight import init_weight
from LOIE_utils.NIQE import calculate_niqe
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from LOIE_utils.Evaluate_utils import calculate_psnr, calculate_mae
from Modules.Dataset import LOIE_Dataset
from net_models.LOIE import LOIE_Net
from Loss.lightenhanceloss import Total_loss

from Modules.Seg_model import *
from pytorch_msssim import ssim


def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def colorize_segmentation(Input_S):
    Input_S = Input_S.squeeze(1)
    colored_output = torch.zeros((Input_S.shape[0], 3, Input_S.shape[1], Input_S.shape[2]), dtype=torch.float32)

    colored_output[:, 0, :, :] = (Input_S == 0).float() * 255
    colored_output[:, 1, :, :] = (Input_S == 1).float() * 255
    colored_output[:, 2, :, :] = (Input_S == 2).float() * 255

    return colored_output

def save_checkpoint(epoch, model, optimizer, scheduler, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path}")

def validate(epoch,val_loader, model, lisudecomp, lisujoint, device):
    lisudecomp.eval()
    lisujoint.eval()
    loss_record = []

    save_dir = '../val_images/val_image'

    psnr_tbar = []
    ssim_tbar = []
    mae_tbar = []
    niqe_tbar = []

    tbar = tqdm(val_loader, desc="Validation")
    with torch.no_grad():
        for i, (L_image, H_image, image_name) in enumerate(tbar):
            L_image = L_image.to(device)
            H_image = H_image.to(device)

            I, R = lisudecomp(L_image)
            Input_joint = torch.cat((I.detach(), R.detach()), dim=1)
            output_lisujoint = lisujoint(Input_joint)
            Input_S = torch.argmax(output_lisujoint, dim=1).float().unsqueeze(1)
            outputs = model(L_image, Input_S)

            batch_size = H_image.shape[0]
            for i in range(batch_size):
                if not os.path.exists(f'{save_dir}/H_result'):
                    os.makedirs(f'{save_dir}/H_result')
                save_image(outputs[i], f'{save_dir}/H_result/{image_name[i].replace("L", "H")}')

                # 调用函数将分割结果转换为红绿蓝三色图像
                colored_segmentation = colorize_segmentation(Input_S[i])

                if not os.path.exists(f'{save_dir}/seg_result/'):
                    os.makedirs(f'{save_dir}/seg_result/')
                save_image(colored_segmentation, f'{save_dir}/seg_result/{image_name[i].replace("L", "S")}')

                if not os.path.exists(f'{save_dir}/L_image'):
                    os.makedirs(f'{save_dir}/L_image')
                save_image(L_image[i], f'{save_dir}/L_image/{image_name[i]}')

                concat_image = torch.concat((L_image[i], outputs[i]), dim=2)
                if not os.path.exists(f'{save_dir}/concat'):
                    os.makedirs(f'{save_dir}/concat')
                save_image(concat_image, f'{save_dir}/concat/{image_name[i].replace("L", "C")}')

                psnr_value = calculate_psnr(H_image[i], outputs[i])
                ssim_value = ssim(H_image[i].unsqueeze(0), outputs[i].unsqueeze(0), data_range=1.0).item()
                mae_value = calculate_mae(H_image[i], outputs[i])
                clean = io.imread(f'{save_dir}/H_result/{image_name[i].replace("L", "H")}')
                niqe_value = calculate_niqe(clean, crop_border=0, params_path='../niqemodels/')

                psnr_tbar.append(psnr_value)
                ssim_tbar.append(ssim_value)
                mae_tbar.append(mae_value)
                niqe_tbar.append(niqe_value)

            val_loss = Total_loss(outputs, H_image)
            loss_record.append(val_loss.item())
            tbar.set_description('Epoch:{0} ValLoss: {1:.3} niqe: {2:.5} psnr: {3:.5} ssim:{4:.5} mae:{5:.5}'.format(epoch,np.mean(loss_record), np.mean(niqe_tbar),np.mean(psnr_tbar), np.mean(ssim_tbar), np.mean(mae_tbar)))

    model.train()
    return np.mean(loss_record)


# Function to plot and save losses
def plot_and_save_loss(train_losses, val_losses, epoch, save_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, 'b', label='Train Loss')
    plt.plot(epochs, val_losses, 'r', label='Val Loss')
    plt.title(f'Train and Validation Loss at Epoch {epoch}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the figure
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(f"{save_dir}/loss_epoch_{epoch}.png")
    plt.close()

def load_checkpoint(model, optimizer, scheduler, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded, starting from epoch {start_epoch}")

    return start_epoch

def  train_and_eval(state_path = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch is running on GPU: {torch.cuda.get_device_name(0)}")
    save_dir = '../train_images/train_image'

    early_stopping = EarlyStopping(patience=10, min_delta=0.001, save_path='../best_model.pth')

    net = LOIE_Net().to(device)
    if state_path == None:
        init_weight(net, nn.init.kaiming_normal_, nn.BatchNorm2d, 1e-3, 0.1, mode='fan_in')
    else:
        net.load_state_dict(torch.load(state_path))
    net.train()

    lisudecomp = LISU_DECOMP().cuda()
    lisujoint = LISU_JOINT().cuda()
    checkpoint = torch.load('../LOIE_utils/Segmodel_checkpoint/164.pth.tar')
    lisudecomp.load_state_dict(checkpoint['state_dict_decomp'])
    lisujoint.load_state_dict(checkpoint['state_dict_enhance'])
    lisudecomp.eval()
    lisujoint.eval()

    batch_size = 2
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])
    train_data = LOIE_Dataset('../data/train/train_H', '../data/train/train_L', transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    val_data = LOIE_Dataset('../data/VAL/L_images', '../data/VAL/GT_images', transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)

    train_losses = []
    val_losses = []

    epochs =  500
    for epoch in range(epochs):
        loss_record = []
        psnr_tbar = []
        ssim_tbar = []
        mae_tbar = []
        niqe_tbar = []

        tbar = tqdm(train_loader)
        for i, (L_image, H_image,image_name) in enumerate(tbar):
            L_image = L_image.to(device)
            H_image = H_image.to(device)
            optimizer.zero_grad()
            I, R = lisudecomp(L_image)
            Input_joint = torch.cat((I.detach(), R.detach()), dim=1)
            output_lisujoint  = lisujoint(Input_joint)
            Input_S = torch.argmax(output_lisujoint, dim=1).float()
            Input_S = Input_S.unsqueeze(1)
            outputs = net(L_image, Input_S)
            batch_size = H_image.shape[0]
            for i in range(batch_size):
                if not os.path.exists(f'{save_dir}/H_result'):
                    os.makedirs(f'{save_dir}/H_result')
                save_image(outputs[i], f'{save_dir}/H_result/{image_name[i].replace("L", "H")}')
                colored_segmentation = colorize_segmentation(Input_S[i])

                if not os.path.exists(f'{save_dir}/seg_result/'):
                    os.makedirs(f'{save_dir}/seg_result/')
                save_image(colored_segmentation, f'{save_dir}/seg_result/{image_name[i].replace("L", "S")}')

                if not os.path.exists(f'{save_dir}/L_image'):
                    os.makedirs(f'{save_dir}/L_image')
                save_image(L_image[i], f'{save_dir}/L_image/{image_name[i]}')

                concat_image = torch.concat((L_image[i], outputs[i]), dim=2)
                if not os.path.exists(f'{save_dir}/concat'):
                    os.makedirs(f'{save_dir}/concat')
                save_image(concat_image, f'{save_dir}/concat/{image_name[i].replace("L", "C")}')

                psnr_value = calculate_psnr(H_image[i], outputs[i])
                ssim_value = ssim(H_image[i].unsqueeze(0), outputs[i].unsqueeze(0), data_range=1.0).item()
                mae_value = calculate_mae(H_image[i], outputs[i])
                clean = io.imread(f'{save_dir}/H_result/{image_name[i].replace("L", "H")}')
                niqe_value = calculate_niqe(clean, crop_border=0, params_path='../niqemodels/')

                psnr_tbar.append(psnr_value)
                ssim_tbar.append(ssim_value)
                mae_tbar.append(mae_value)
                niqe_tbar.append(niqe_value)

            loss = Total_loss(outputs, H_image)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())
            tbar.set_description('Epoch:{0} TrainLoss: {1:.3} niqe: {2:.5} psnr: {3:.5} ssim:{4:.5} mae:{5:.5}'.format(epoch,np.mean(loss_record), np.mean(niqe_tbar),np.mean(psnr_tbar), np.mean(ssim_tbar), np.mean(mae_tbar)))

        scheduler.step()
        save_path = f"../weights/save_LOIE_weights/Epoch{epoch+1}.pth"
        save_checkpoint(epoch, net, optimizer, scheduler, save_path)

        train_loss = np.mean(loss_record)
        val_loss = validate(epoch,val_loader,net,lisudecomp, lisujoint, device)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        plot_and_save_loss(train_losses, val_losses, epoch + 1, '../loss_images/save_loss_image')
        early_stopping(val_loss,net)
        if early_stopping.early_stop:
            print("Early stopping")
            break

if __name__ == "__main__":
    train_and_eval()