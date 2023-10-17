import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ContrastiveAugmentedDataSet(Dataset):
    def __init__(self, data_arr: np.array, transform):
        self.transform = transform
        self.data_arr = data_arr

    def __len__(self):
        return self.data_arr.shape[0]

    def __getitem__(self, idx):
        img = self.data_arr[idx]
        # Generate two different augmentations for the same image
        print(f"img.shape: {img.shape}")
        aug_img1, aug_img2 = self.transform(img), self.transform(img)
        return aug_img1, aug_img2


def get_contrastive_augmentation(patch_size=(256, 256)):
    transform = T.Compose([
                           T.RandomResizedCrop(patch_size, scale=(0.8, 1.0)),
                           T.RandomHorizontalFlip(p=0.5),
                           T.ToTensor(),
                           #T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
                           ])
    return transform