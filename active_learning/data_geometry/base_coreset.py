import h5py
from glob import glob
from scipy.ndimage.interpolation import zoom
import numpy as np
from tqdm import tqdm
from numpy.random import RandomState
import os
from active_learning.data_geometry.base_data_geometry import BaseDataGeometry
from active_learning.data_geometry.feature_model_factory import FeatureModelFactory
from active_learning.data_geometry import coreset_algs


class BaseCoreset(BaseDataGeometry):
    """Base class for Coreset sampling"""

    def __init__(self, alg_string="kcenter_greedy", metric='euclidean', patch_size=(256, 256), feature_model=False,
                 feature_model_params=None, contrastive=False, use_model_features=False, seed=0, gpus="cuda:0",
                 **kwargs):
        super().__init__()
        self.alg_string = alg_string
        self.metric = metric
        print(f"Using the {self.metric} metric in Coreset sampling...")
        self.patch_size = patch_size
        self.contrastive = contrastive
        self.gpus = gpus
        if feature_model_params is None:
            feature_model_params = {}
        self.feature_model = FeatureModelFactory.create_feature_model(model=feature_model, contrastive=self.contrastive,
                                                                      gpus=self.gpus, **feature_model_params)
        self.use_model_features = use_model_features
        self.seed = seed
        self.random_state = RandomState(seed=self.seed)
        self.basic_coreset_alg = None
        self.data_root = None
        self.all_train_im_files = None
        self.all_train_full_im_paths = None
    
    def setup(self, data_root, all_train_im_files):
        self.setup_data(data_root, all_train_im_files)
        self.setup_alg()

    def setup_data(self, data_root, all_train_im_files):
        print("Initializing Training pool X for coreset sampling!")
        self.data_root = data_root
        self.all_train_im_files = all_train_im_files
        self.all_train_full_im_paths = [os.path.join(data_root, im_path) for im_path in all_train_im_files]
        self.setup_image_features()

    def setup_image_features(self):
        image_data = self._get_data()
        self.feature_model.init_image_features(image_data)

    def setup_alg(self):
        if self.alg_string in coreset_algs:
            self.coreset_cls = coreset_algs[self.alg_string]
        else:
            raise ValueError(f"No coreset alg found for {self.alg_string}")

    def get_features(self):
        return self.feature_model.get_features()

    def create_coreset_inst(self, processed_data):
        return self.coreset_cls(processed_data, metric=self.metric, seed=self.seed)

    def get_coreset_inst_and_features_for_round(self, round_dir, train_logits_path, delete_preds=True):
        if self.use_model_features:
            print("Using Model Features")
            feat = self.get_model_features(round_dir, train_logits_path, delete_preds=delete_preds)
            if feat is None:
                print("Model features not found. Using image features instead")
                feat = self.get_features()
            coreset_inst = self.create_coreset_inst(feat)
        else:
            feat = self.get_features()
            print("Finished getting model features")
            coreset_inst = self.create_coreset_inst(feat)
        return coreset_inst, feat

    def get_model_features(self, prev_round_dir, train_logits_path, delete_preds=True):
        if prev_round_dir is None:
            print("No model features for first round!")
            return None
        train_logits_path = os.path.join(prev_round_dir, "*", train_logits_path)
        print(f"Looking for model features in {train_logits_path}")
        train_results = sorted(list(glob(train_logits_path)))
        if len(train_results) == 0:
            raise ValueError("No model features found!")
        if len(train_results) > 1:
            raise ValueError(f"More than one prediction file found: {train_results}")
        train_results = train_results[0]
        # useful for how to load npz (using "incorrect version): https://stackoverflow.com/questions/61985025/numpy-load-part-of-npz-file-in-mmap-mode
        preds_arrs = []
        for im_file in tqdm(self.all_train_im_files):
            preds_arr = np.load(train_results, mmap_mode='r')[os.path.basename(im_file)]
            preds_arrs.append(preds_arr)
        preds_arrs = np.stack(preds_arrs, axis=0)
        # flatten array except for first dim
        preds_arrs = preds_arrs.reshape(preds_arrs.shape[0], -1)

        # after obtaining features, delete the *.npz files for the round
        if delete_preds:
            os.remove(train_results)

        return preds_arrs


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

