from abc import abstractmethod
import os
from scipy.ndimage.interpolation import zoom
import numpy as np 
import h5py
from torch.utils.data import Dataset
import torchvision.transforms as T
from active_learning.data_geometry.base_data_geometry import BaseDataGeometry


class BaseAdversarial(BaseDataGeometry):
    """Base class for Adversarial sampling"""

    def __init__(self, latent_dim=32, batch_size=128, num_epochs=50, patch_size=(256, 256),
                 gpus="cuda:0", use_model_features=False, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patch_size = patch_size
        self.gpus = gpus
        self.use_model_features = use_model_features
        
    def setup(self, exp_dir, data_root, all_train_im_files):
        self.setup_data(data_root, all_train_im_files)

    def setup_data(self, data_root, all_train_im_files):
        print("Initializing Training pool X for VAAL sampling!")
        self.data_root = data_root
        self.all_train_im_files = all_train_im_files
        self.all_train_full_im_paths = [os.path.join(data_root, im_path) for im_path in all_train_im_files]
        self.pool = VAALDatasetWrapper(self.all_train_full_im_paths, transform=T.ToTensor(),
                                      patch_size=self.patch_size)
        print("Done setting up data")

    @abstractmethod
    def train_adv_model(self):
        raise NotImplementedError()
    
    
class VAALDatasetWrapper(Dataset):
    def __init__(self, all_train_full_im_paths, transform=None, patch_size=(256, 256)):
        self.sample_list = all_train_full_im_paths
        self.transform = transform
        self.patch_size = patch_size
        
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        case = self.sample_list[idx]
        with h5py.File(case, 'r') as hf:
            sample = hf['image'][:]
        sample = self._patch_im(sample, self.patch_size)
        if self.transform:
            sample = self.transform(sample)
        return sample, idx
    
    @staticmethod
    def _patch_im(im, patch_size):
        x, y = im.shape
        image = zoom(im, (patch_size[0] / x, patch_size[1] / y), order=0)
        return image