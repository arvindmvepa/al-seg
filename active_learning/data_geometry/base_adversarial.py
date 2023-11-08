from abc import abstractmethod
import os
from active_learning.data_geometry.base_data_geometry import BaseDataGeometry


class BaseAdversarial(BaseDataGeometry):
    """Base class for Adversarial sampling"""

    def __init__(self, latent_dim=32, batch_size=128, num_epochs=100, gpus="cuda:0", **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.gpus = gpus

    def setup(self, data_root, all_train_im_files):
        self.data_root = data_root
        self.all_train_im_files = all_train_im_files
        self.all_train_full_im_paths = [os.path.join(data_root, im_path) for im_path in all_train_im_files]

    @abstractmethod
    def train_adv_model(self):
        raise NotImplementedError()