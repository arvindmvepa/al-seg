from abc import ABC, abstractmethod
import h5py 
from scipy.ndimage.interpolation import zoom
import numpy as np
from tqdm import tqdm


class BaseCoreset(ABC):
    """Abstract class for Coreset sampling"""

    def __init__(self, im_path_list=None, patch_size=(256, 256), **kwargs):
        self.im_path_list = im_path_list
        self.patch_size = patch_size
        self.X = None

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

    def _get_data(self, im_path_list):
        cases = []
        for im_path in tqdm(im_path_list):
            image = self._load_image(im_path)
            cases.append(image)
        cases_arr = np.concatenate(cases, axis=0)
        return cases_arr 
    
    def _init_X(self, im_path_list):
        print("Initializing Training pool X for coreset sampling!")
        self.im_path_list = im_path_list
        self.X = self._get_data(im_path_list)

    @abstractmethod
    def calculate_representativeness(self, im_score_file, round_dir, ignore_ims_dict, skip=False, **kwargs):
        raise NotImplementedError()
    

class NoCoreset(BaseCoreset):
    """Placeholder class for no data representativeness"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pass_ = True

    def calculate_representativeness(self, im_score_file, round_dir, ignore_ims_dict):
        pass