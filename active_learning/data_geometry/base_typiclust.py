from abc import abstractmethod
import os
from numpy.random import RandomState
from active_learning.data_geometry.base_data_geometry import BaseDataGeometry
from active_learning.feature_model.feature_model_factory import FeatureModelFactory
from active_learning.dataset.dataset_factory import DatasetFactory


class BaseTypiclust(BaseDataGeometry):
    """Base class for Adversarial sampling"""

    def __init__(self, patch_size=(224, 224), dataset_type="ACDC", dataset_kwargs=None, gpus="cuda:0",
                 feature_model=False, feature_model_params=None, contrastive=False, use_model_features=False,
                 use_labels=False, seed=0, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.dataset_type = dataset_type
        if not isinstance(dataset_kwargs, dict):
            self.dataset_kwargs = {}
        else:
            self.dataset_kwargs = dataset_kwargs
        self.gpus = gpus
        self.feature_model = feature_model
        self.feature_model_params = feature_model_params
        self.contrastive = contrastive
        self.use_model_features = use_model_features
        self.user_labels = use_labels
        self.seed = seed
        self.random_state = RandomState(seed=self.seed)
        
    def setup(self, exp_dir, data_root, all_train_im_files):
        self.setup_feature_model(exp_dir)
        self.setup_data(data_root, all_train_im_files)

    def setup_feature_model(self, exp_dir):
        print("Setting up feature model...")
        self.exp_dir = exp_dir
        if self.feature_model_params is None:
            self.feature_model_params = {}
        self.feature_model = FeatureModelFactory.create_feature_model(model=self.feature_model,
                                                                      contrastive=self.contrastive,
                                                                      gpus=self.gpus,
                                                                      exp_dir=self.exp_dir,
                                                                      **self.feature_model_params)
        print("Done setting up feature model.")

    def setup_data(self, data_root, all_train_im_files):
        print("Setting up data")
        self.data_root = data_root
        self.all_train_im_files = all_train_im_files
        self.all_train_full_im_paths = [os.path.join(data_root, im_path) for im_path in all_train_im_files]
        self.dataset = DatasetFactory.create_dataset(dataset_type=self.dataset_type,
                                                     all_train_im_files=self.all_train_im_files,
                                                     all_train_full_im_paths=self.all_train_full_im_paths,
                                                     **self.dataset_kwargs)
        self.setup_image_features()
        self.features = self.feature_model.get_features()
        print("Done setting up data")

    def setup_image_features(self):
        print("Setting up image features...")
        print("Getting data")
        image_data, self.image_meta_data, self.image_labels_arr,  =  self.dataset.get_data(self.use_labels)
        print("Processing meta_data...")
        self.image_meta_data_arr = self.dataset.process_meta_data(self.image_meta_data)
        self.non_image_indices = self.dataset.get_non_image_indices()
        print("Initializing image features for feature model...")
        self.feature_model.init_image_features(image_data, self.image_meta_data_arr, self.non_image_indices)
        print("Done setting up image features")
    
    @abstractmethod
    def select_samples(self):
        raise NotImplementedError()