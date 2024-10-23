import os
import numpy as np
from skimage import io
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from Modules.Dataset import LOIE_Test_Dataset
from net_models.LOIE import LOIE_Net
from LOIE_utils.NIQE import calculate_niqe
from Modules.Seg_model import *

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


def  train(state_path = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch is running on GPU: {torch.cuda.get_device_name(0)}")
    save_dir = '../test_result/test_images'
    H_result_dir = f'{save_dir}/H_result'
    L_image_dir = f'{save_dir}/L_image'
    seg_result_dir = f'{save_dir}/seg_result'
    concat_dir = f'{save_dir}/concat'
    dir_list = [H_result_dir, L_image_dir,seg_result_dir,concat_dir]
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)

    lisudecomp = LISU_DECOMP().cuda()
    lisujoint = LISU_JOINT().cuda()
    checkpoint = torch.load('../LOIE_utils/Segmodel_checkpoint/164.pth.tar')
    lisudecomp.load_state_dict(checkpoint['state_dict_decomp'])
    lisujoint.load_state_dict(checkpoint['state_dict_enhance'])
    lisudecomp.eval()
    lisujoint.eval()

    #创建网络实例
    net = LOIE_Net().to(device)
    checkpoint_enhance = torch.load(state_path)
    net.load_state_dict(checkpoint_enhance)

    #数据集
    batch_size = 1
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256)),
    ])
    test_data = LOIE_Test_Dataset('../data/test/test_L', transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        for epoch in range(1):
            niqe_tbar = []

            tbar = tqdm(test_loader)
            for i, (L_image,image_name) in enumerate(tbar):
                L_image = L_image.to(device)
                image_name = ''.join(image_name)

                I, R = lisudecomp(L_image)
                Input_joint = torch.cat((I.detach(), R.detach()), dim=1)
                Input_S = torch.argmax(lisujoint(Input_joint), dim=1).float()
                Input_S = Input_S.unsqueeze(1)
                outputs = net(L_image, Input_S)
                image_name_H = image_name.replace('L', 'L_images')

                save_image(outputs,f'{H_result_dir}/{image_name_H}')
                save_image(L_image, f'{L_image_dir}/{image_name}')
                colored_segmentation = colorize_segmentation(Input_S)
                image_name_seg = image_name.replace('L', 'S')
                save_image(colored_segmentation, f'{seg_result_dir}/{image_name_seg}')
                concat = torch.cat((L_image, outputs), dim=2)
                image_name_c = image_name.replace('L', 'C')
                save_image(concat, f'{concat_dir}/{image_name_c}')
                clean = io.imread(f'{H_result_dir}/{image_name_H}')
                niqe_value = calculate_niqe(clean, crop_border=0, params_path='../niqemodels/')
                niqe_tbar.append(niqe_value)
                tbar.set_description('niqe: {0:.5}'.format(np.mean(niqe_tbar)))


if __name__ == "__main__":
    train('../best_model.pth')