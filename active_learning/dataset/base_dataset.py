from abc import ABC, abstractmethod
import h5py
from tqdm import tqdm
import numpy as np
from scipy.ndimage.interpolation import zoom


class BaseDataset(ABC):
    """Base class for Dataset"""

    def __init__(self, all_train_im_files, all_train_full_im_paths, **kwargs):
        super().__init__()
        self.all_train_im_files = all_train_im_files
        self.all_train_full_im_paths = all_train_full_im_paths

    def get_data(self, all_train_full_im_paths, use_labels=False):
        if use_labels:
            cases_arr, labels_arr, meta_data = self._get_image_and_label_data(all_train_full_im_paths)
        else:
            labels_arr = None
            cases_arr, meta_data = self._get_image_data(all_train_full_im_paths)
        return cases_arr, labels_arr, meta_data

    @staticmethod
    def _patch_im(im, patch_size):
        x, y = im.shape
        image = zoom(im, (patch_size[0] / x, patch_size[1] / y), order=0)
        return image

    def _load_image(self, case):
        h5f = h5py.File(case, 'r')
        image = h5f['image'][:]
        patched_image = self._patch_im(image, self.patch_size)
        patched_image = patched_image[np.newaxis,]
        return patched_image, None

    def _load_image_and_label(self, case):
        h5f = h5py.File(case, 'r')
        image = h5f['image'][:]
        patched_image = self._patch_im(image, self.patch_size)
        patched_image = patched_image[np.newaxis,]
        label = h5f[self.ann_type][:]
        patched_label = self._patch_im(label, self.patch_size)
        return patched_image, patched_label[np.newaxis,]

    def _load_meta_data(self, meta_data_file):
        meta_data = {}
        with open(meta_data_file, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                if key == 'ED' or key == 'ES':
                    value = int(value)
                if key == 'Height' or key == 'Weight':
                    value = float(value)
                meta_data[key] = value
        return meta_data

    def _get_image_data(self, all_train_full_im_paths):
        cases = []
        meta_data = []
        for im_path in tqdm(all_train_full_im_paths):
            image = self._load_image(im_path)
            meta_datum = self._load_meta_data(im_path)
            cases.append(image)
            meta_data.append(meta_datum)
        cases_arr = np.concatenate(cases, axis=0)
        return cases_arr, meta_data

    def _get_image_and_label_data(self, all_train_full_im_paths):
        cases = []
        labels = []
        meta_data = []
        for im_path in tqdm(all_train_full_im_paths):
            image, label = self._load_image_and_label(im_path)
            meta_datum = self._load_meta_data(im_path)
            cases.append(image)
            labels.append(label)
            meta_data.append(meta_datum)
        cases_arr = np.concatenate(cases, axis=0)
        labels_arr = np.concatenate(labels, axis=0)
        return cases_arr, labels_arr, meta_data

    @abstractmethod
    def _load_meta_data(self, im_data_file):
        raise NotImplementedError()

    @abstractmethod
    def process_meta_data(self):
        raise NotImplementedError()

    @abstractmethod
    def get_non_image_indices(self):
        raise NotImplementedError()
