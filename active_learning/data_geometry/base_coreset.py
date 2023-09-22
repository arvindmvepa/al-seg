import h5py
from scipy.ndimage.interpolation import zoom
import numpy as np
from tqdm import tqdm
import os
from active_learning.data_geometry.base_data_geometry import BaseDataGeometry
from active_learning.data_geometry import coreset_algs


class BaseCoreset(BaseDataGeometry):
    """Base class for Coreset sampling"""

    def __init__(self, alg_string, patch_size=(256, 256), **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.alg_string = alg_string
        self.coreset_alg = None
        self.data_root = None
        self.all_train_im_files = None
        self.all_train_full_im_paths = None
        self.all_processed_train_data = None

    @staticmethod
    def _patch_im(im, patch_size):
        x, y = im.shape
        image = zoom(im, (patch_size[0] / x, patch_size[1] / y), order=0)
        return image 

    def _load_image(self, case):
        h5f = h5py.File(case, 'r')
        image = h5f['image'][:]
        patched_image = self._patch_im(image, self.patch_size)
        return patched_image[np.newaxis,]

    def _get_data(self):
        cases = []
        for im_path in tqdm(self.all_train_full_im_paths):
            image = self._load_image(im_path)
            cases.append(image)
        cases_arr = np.concatenate(cases, axis=0)
        return cases_arr 
    
    def setup(self, data_root, all_train_im_files):
        print("Initializing Training pool X for coreset sampling!")
        self.data_root = data_root
        self.all_train_im_files = all_train_im_files
        self.all_train_full_im_paths = [os.path.join(data_root, im_path) for im_path in all_train_im_files]
        all_processed_train_data = self._get_data()
        self.coreset_alg = coreset_algs[self.alg_string](all_processed_train_data)