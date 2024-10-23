import os
import numpy as np
from skimage import io
from LOIE_utils.NIQE import calculate_niqe

list_dir = ['../data/niqe_pinggu/Ours/DICM/H_result',
            '../data/niqe_pinggu/Ours/LIME/H_result',
            '../data/niqe_pinggu/Ours/MEF/H_result',
            '../data/niqe_pinggu/Ours/NPE/H_result',
            '../data/niqe_pinggu/Ours/VV/H_result',]

for dir in list_dir:
    list_images = os.listdir(dir)
    folder_name = dir.split('/')[-2]
    niqe = []
    for image in list_images:
            image_niqe = io.imread(os.path.join(dir, image))
            niqe_value = calculate_niqe(image_niqe, crop_border=0, params_path='../niqemodels/')
            niqe.append(niqe_value)
    print(f'{folder_name}的NIQE是{np.mean(niqe):.2f}')