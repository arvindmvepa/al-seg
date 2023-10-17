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
    transform = T.Compose([T.ToTensor(),
                           T.RandomResizedCrop(patch_size, scale=(0.8, 1.0)),
                           T.RandomHorizontalFlip(p=0.5),
                           T.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
                           T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
                           T.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
                           ])
    return transform