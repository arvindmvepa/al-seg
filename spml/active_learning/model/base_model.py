from abc import ABC, abstractmethod
from spml.active_learning.model.model_params import model_params
import json
import os
import time


class BaseModel(ABC):
    """Abstract class for model interface

    Creates an interface for the model to interact with different object
    types, including policy and uncertainty objects

    Attributes
    ----------
    model_params : dict
    ann_type : str
    data_root : str
    ensemble_size : int
    epoch_len : int
    num_epochs : int
    seed : int
    tag : str
    virtualenv : str

    Methods
    -------
    train_model(model_no, snapshot_dir, round_dir,
    cur_total_oracle_split=0, cur_total_pseudo_split=0,cuda_devices="0",
    save_params=None)
        Trains an instance of the model from scratch
    train_ensemble(self, round_dir, cur_total_oracle_split=0,
    cur_total_pseudo_split=0, cuda_devices="0", skip=False,
    save_params=None)
        Trains an ensemble of the models from scratch
    get_ensemble_scores(score_func, im_score_file, round_dir, ignore_ims_dict)
        Generates scores from the ensemble predictions
    get_round_train_file_paths(round_dir, cur_total_oracle_split, **kwargs):
        Generates file paths for data corresponding to `file_keys`

    """

    def __init__(self, ann_type="box", data_root=".", ensemble_size=1,  epoch_len = 10578, num_epochs = 3,
                 seed=0, tag="", virtualenv='/home/asjchoi/SPML_Arvind/spml-env'):
        self.model_params = model_params[self.model_string][ann_type]
        self.ann_type = ann_type
        self.data_root = data_root
        self.ensemble_size = ensemble_size
        self.epoch_len = epoch_len
        self.num_epochs = num_epochs
        self.seed = seed
        self.tag = tag
        self.virtualenv = virtualenv
        if len(self.file_keys) == 0:
            raise ValueError(f"file_keys needs at least one key")
        self._init_train_file_info()

    @abstractmethod
    def train_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0,
                    cuda_devices="0", save_params=None):
        raise NotImplementedError()

    def train_ensemble(self, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0, cuda_devices="0",
                       skip=False, save_params=None):
        if skip:
            print("Skip Training Ensemble")
            return
        print("Start Training Ensemble")
        for model_no in range(self.ensemble_size):
            snapshot_dir = os.path.join(round_dir, str(model_no))
            if not os.path.exists(snapshot_dir):
                os.makedirs(snapshot_dir)
            self.train_model(model_no=model_no, snapshot_dir=snapshot_dir, round_dir=round_dir,
                             cur_total_oracle_split=cur_total_oracle_split,
                             cur_total_pseudo_split=cur_total_pseudo_split, cuda_devices=cuda_devices,
                             save_params=save_params)
        print("Finished Training Ensemble")

    def get_ensemble_scores(self, score_func, im_score_file, round_dir, ignore_ims_dict):
        raise NotImplementedError()

    @abstractmethod
    def _init_train_file_info(self):
        raise NotImplementedError()

    @abstractmethod
    def get_round_train_file_paths(self, round_dir, cur_total_oracle_split, **kwargs):
        raise NotImplementedError()

    def train_split(self, cur_total_oracle_split, cur_total_pseudo_split):
        return f"{self.model_params['train_split']}_o{cur_total_oracle_split}_p{cur_total_pseudo_split}_seed{self.seed}"

    @property
    @abstractmethod
    def file_keys(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def im_key(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def model_string(self):
        raise NotImplementedError()

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError()
