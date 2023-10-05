import h5py
from scipy.ndimage.interpolation import zoom
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch
import torch.nn as nn
from numpy.random import RandomState
import os
from torchvision.models import resnet18, resnet50
from active_learning.data_geometry.base_data_geometry import BaseDataGeometry
from active_learning.data_geometry import coreset_algs


class BaseCoreset(BaseDataGeometry):
    """Base class for Coreset sampling"""

    def __init__(self, alg_string="kcenter_greedy", metric='euclidean', patch_size=(256, 256), feature_model=None,
                 feature_model_ignore_layer=-1, feature_model_batch_size=128, seed=0, gpus="cuda:0", **kwargs):
        super().__init__()
        self.alg_string = alg_string
        self.metric = metric
        print(f"Using the {self.metric} metric in Coreset sampling...")
        self.patch_size = patch_size
        self.feature_model_batch_size = feature_model_batch_size
        self.seed = seed
        self.gpus = gpus
        self.random_state = RandomState(seed=self.seed)
        if feature_model == 'resnet18':
            print("Using Resnet18 for feature extraction...")
            self.feature_model = resnet18(pretrained=True)
        elif feature_model == 'resnet50':
            print("Using Resnet50 for feature extraction...")
            self.feature_model = resnet50(pretrained=True)
        else:
            self.feature_model = None
        if self.feature_model is not None:
            # only layers before feature_model_ignore_layer will be used for feature extraction
            self.feature_model = nn.Sequential(*list(self.feature_model.children())[:feature_model_ignore_layer])
            self.feature_model = self.feature_model.to(self.gpus)
            self.feature_model.eval()
        self.basic_coreset_alg = None
        self.data_root = None
        self.all_train_im_files = None
        self.all_train_full_im_paths = None
        self.image_features = None
    
    def setup(self, data_root, all_train_im_files):
        self.setup_data(data_root, all_train_im_files)
        self.setup_alg()

    def setup_data(self, data_root, all_train_im_files):
        print("Initializing Training pool X for coreset sampling!")
        self.data_root = data_root
        self.all_train_im_files = all_train_im_files
        self.all_train_full_im_paths = [os.path.join(data_root, im_path) for im_path in all_train_im_files]
        self.image_features =  self._get_data()
        if self.feature_model is not None:
            print("Extracting features for all training data using feature_model...")
            dataset = CoresetDatasetWrapper(self.image_features, transform=T.ToTensor())
            alL_data_dataloader = DataLoader(dataset, batch_size=self.feature_model_batch_size,
                                             shuffle=False, pin_memory=True)
            self.image_features = self.get_nn_features(alL_data_dataloader)

    def setup_alg(self):
        if self.alg_string in coreset_algs:
            self.coreset_cls = coreset_algs[self.alg_string]
            self.basic_coreset_alg = coreset_algs[self.alg_string](self.image_features,
                                                                   metric=self.metric,
                                                                   seed=self.seed)
        else:
            print(f"No coreset alg found for {self.alg_string}")
            self.coreset_cls = None
            self.basic_coreset_alg = None

    def create_coreset_inst(self, processed_data):
        return self.coreset_cls(processed_data, metric=self.metric, seed=self.seed)

    def get_nn_features(self, data_loader):
        features = torch.tensor([]).to(self.gpus)
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs.to(self.gpus)
                # convert from grayscale to color, hard-coded for pretrained resnet
                inputs = torch.cat([inputs, inputs, inputs], dim=1)
                # flatten the feature map (but not the batch dim)
                features_batch = self.feature_model(inputs).flatten(1)
                features = torch.cat((features, features_batch), 0)
            feat = features
        return feat.detach().cpu().numpy()

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
        # flatten array except for first dim
        cases_arr = cases_arr.reshape(cases_arr.shape[0], -1)
        return cases_arr


class CoresetDatasetWrapper(Dataset):

    def __init__(self, image_features, transform=None):
        self.image_features = image_features
        self.transform = transform

    def __len__(self):
        return len(self.image_features)

    def __getitem__(self, idx):
        sample = self.image_features[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
