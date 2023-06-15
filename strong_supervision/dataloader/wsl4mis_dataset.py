import os

import h5py

import torch
from torch.utils.data import Dataset


class WSL4MISDataset(Dataset):
    """WSL4MIS dataset."""
    def __init__(self, split='train', transform=None, train_file="train.txt", 
                 val_file="val.txt", root_dir="."):
        
        self._root_dir = root_dir
        self.sample_list =  []
        self.split = split
        self.transform = transform
        self.train_path = None
        self.val_path = None

        if self.split == 'train':
            self.train_path = os.path.join(self._root_dir, train_file)
            self.all_slices = os.listdir(self.train_path)
            self.sample_list = self.all_slices
        
        elif self.split == 'val':
            self.val_path = os.path.join(self._root_dir, val_file)
            self.all_volumes = os.listdir(self.val_path)
            self.sample_list = self.all_volumes

        else:
            raise ValueError("split must be either 'train' or 'val'")
        
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        im = self.sample_list[idx]
        if self.split == 'train':
            h5f = h5py.File(os.path.join(self.train_path, im), 'r')
        else:
            h5f = h5py.File(os.path.join(self.val_path, im), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        h5f.close()

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample 
