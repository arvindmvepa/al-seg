import os
import random
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import h5py
import re

import torch
from torch.utils.data import Dataset

class WSL4MISDataset(Dataset):
    """WSL4MIS dataset."""
    def __init__(self, split='train', data_dir=".", fold='fold1', transform=None):
        self._data_dir = data_dir
        self.sample_list =  []
        self.split = split
        self.transform = transform
        train_ids, test_ids = self._get_fold_ids(fold)
        if self.split == 'train':
            self.all_slices = os.listdir(
                os.path.join(self._data_dir, "ACDC_training_slices"))
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)
                
        elif self.split == 'val':
            self.all_volumes = os.listdir(
                os.path.join(self._data_dir, "ACDC_training_volumes"))
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        else:
            raise ValueError("split must be either 'train' or 'val'")
        
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        fold1_testing_set = [
            "patient{:0>3}".format(i) for i in range(1, 21)]
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>3}".format(i) for i in range(41, 61)]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            "patient{:0>3}".format(i) for i in range(61, 81)]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        fold5_testing_set = [
            "patient{:0>3}".format(i) for i in range(81, 101)]
        fold5_training_set = [
            i for i in all_cases_set if i not in fold5_testing_set]
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        im = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._data_dir +
                            "/ACDC_training_slices/{}".format(im), 'r')
        else:
            h5f = h5py.File(self._data_dir +
                            "/ACDC_training_volumes/{}".format(im), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        h5f.close()

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample 


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            if 4 in np.unique(label):
                image, label = random_rotate(image, label, cval=4)
            else:
                image, label = random_rotate(image, label, cval=0)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample
    

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    return image, label