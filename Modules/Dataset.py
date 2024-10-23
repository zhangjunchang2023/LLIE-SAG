import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('L', 'S'))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        image = np.array(image)
        mask = np.array(mask)/255

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return (image, mask)

class LOIE_Dataset(Dataset):
    def __init__(self, L_image_dir, H_image_dir, transform=None):
        self.L_image_dir= L_image_dir
        self.H_image_dir = H_image_dir
        self.transform = transform
        self.images = os.listdir(L_image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        L_image_path = os.path.join(self.L_image_dir, img_name)
        H_image_path = os.path.join(self.H_image_dir , img_name.replace('L', 'H'))
        L_image = Image.open(L_image_path).convert("RGB")
        H_image = Image.open(H_image_path).convert("RGB")

        if self.transform:
            L_image = self.transform(L_image)
            H_image = self.transform(H_image)

        return (L_image, H_image,img_name)

class LOIE_Test_Dataset(Dataset):
    def __init__(self, L_image_dir,transform=None):
        self.L_image_dir = L_image_dir
        self.transform = transform
        self.images = os.listdir(L_image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        L_image_path = os.path.join(self.L_image_dir, img_name)
        L_image = Image.open(L_image_path).convert("RGB")

        if self.transform:
            L_image = self.transform(L_image)

        return (L_image,img_name)

