from abc import ABC, abstractmethod
from active_learning.model.model_params import model_params
import os
from random import Random
from tqdm import tqdm
import numpy as np
from glob import glob
import shutil
from PIL import Image

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

    def __init__(self, ann_type="box", data_root=".", ensemble_size=1,  seed=0, gpus="0", tag=""):
        self.model_params = model_params[self.model_string][ann_type]
        self.ann_type = ann_type
        self.data_root = data_root
        self.ensemble_size = ensemble_size
        self.seed = seed
        self.random_gen = Random(seed)
        self.gpus = gpus
        self.tag = tag
        if len(self.file_keys) == 0:
            raise ValueError(f"file_keys needs at least one key")
        self.all_train_files_dict = None
        self._init_train_file_info()
        self._init_val_file_info()
        self._init_test_file_info()

    @abstractmethod
    def train_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        raise NotImplementedError()
    
    @abstractmethod
    def inf_train_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        raise NotImplementedError()
    
    @abstractmethod
    def inf_val_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        raise NotImplementedError()
    
    def inf_test_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        raise NotImplementedError()
    
    def inf_train(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        self.inf_train_model(model_no=model_no, snapshot_dir=snapshot_dir, round_dir=round_dir,
                             cur_total_oracle_split=cur_total_oracle_split, 
                             cur_total_pseudo_split=cur_total_pseudo_split)

    def inf_val(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        self.inf_val_model(model_no=model_no, snapshot_dir=snapshot_dir, round_dir=round_dir,
                           cur_total_oracle_split=cur_total_oracle_split,
                           cur_total_pseudo_split=cur_total_pseudo_split)
    
    def inf_test(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        self.inf_test_model(model_no=model_no, snapshot_dir=snapshot_dir, round_dir=round_dir,
                            cur_total_oracle_split=cur_total_oracle_split,
                            cur_total_pseudo_split=cur_total_pseudo_split)

    def train_ensemble(self, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0, skip=False, 
                       train=True, inf_train=False, inf_val=True, inf_test=False):
        if skip:
            print("Skip Training Ensemble")
            return
        print("Start Training Ensemble")
        for model_no in range(self.ensemble_size):
            snapshot_dir = os.path.join(round_dir, str(model_no))
            if not os.path.exists(snapshot_dir):
                os.makedirs(snapshot_dir)
            if train:
                self.train_model(model_no=model_no, snapshot_dir=snapshot_dir, round_dir=round_dir, 
                                 cur_total_oracle_split=cur_total_oracle_split, 
                                 cur_total_pseudo_split=cur_total_pseudo_split)
            if inf_train:
                self.inf_train(model_no=model_no, snapshot_dir=snapshot_dir, round_dir=round_dir, 
                               cur_total_oracle_split=cur_total_oracle_split, 
                               cur_total_pseudo_split=cur_total_pseudo_split)
            if inf_val: 
                self.inf_val(model_no=model_no, snapshot_dir=snapshot_dir, round_dir=round_dir, 
                             cur_total_oracle_split=cur_total_oracle_split, 
                             cur_total_pseudo_split=cur_total_pseudo_split)
            if inf_test:
                self.inf_test(model_no=model_no, snapshot_dir=snapshot_dir, round_dir=round_dir, 
                              cur_total_oracle_split=cur_total_oracle_split, 
                              cur_total_pseudo_split=cur_total_pseudo_split)
        print("Finished Training Ensemble")

    def get_ensemble_scores(self, score_func, im_score_file, round_dir, ignore_ims_dict, delete_preds=True):
        raise NotImplementedError()

    @abstractmethod
    def _init_train_file_info(self):
        raise NotImplementedError()

    @abstractmethod
    def _init_val_file_info(self):
        raise NotImplementedError()

    def _init_test_file_info(self):
        pass

    @abstractmethod
    def get_round_train_file_paths(self, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
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


class MajorityVoteMixin:

    def get_ensemble_scores(self, score_func, im_score_file, round_dir, ignore_ims_dict, delete_preds=True):
        f = open(im_score_file, "w")
        train_results_dirs = sorted(list(glob(os.path.join(round_dir, "*",
                                                           self.model_params['train_results_dir']))))
        filt_models_result_files = self._filter_unann_ims(train_results_dirs, ignore_ims_dict)
        for models_result_file in tqdm(zip(*filt_models_result_files)):
            results_arr, base_name = self._convert_ensemble_results_to_arr(models_result_file)
            # calculate the score_func over the ensemble of predictions
            score = score_func(results_arr)
            f.write(f"{base_name},{np.round(score, 7)}\n")
            f.flush()
        f.close()
        # after obtaining scores, delete train results for the rounds
        if delete_preds:
            for train_result_dir in train_results_dirs:
                shutil.rmtree(train_result_dir)

    def _convert_ensemble_results_to_arr(self, models_result_file):
        results = []
        for model_result_file in models_result_file:
            arr = np.asarray(Image.open(model_result_file).convert('L'))
            results.append(arr)
        results_arr = np.stack(results)
        # use the first model's pred_file basename because it's the same image
        base_name = os.path.basename(models_result_file[0])
        return results_arr, base_name

    def _filter_unann_ims(self, train_results_dirs, ignore_ims_dict):
        # load models' results files
        models_result_files = [sorted(glob(os.path.join(train_results_dir, "*"))) for train_results_dir in
                               train_results_dirs]
        # filter model results in which we already have annotations
        # Note: filter predictions for first model and use the filtering results for the other models
        filt_models_result_files = []
        for _ in range(self.ensemble_size):
            filt_models_result_files.append([])
        for i, result_file in enumerate(models_result_files[0]):
            remove = False
            for im_file in ignore_ims_dict[self.im_key]:
                if os.path.basename(result_file)[:-4] in im_file:
                    remove = True
                if remove:
                    break
            if not remove:
                for j in range(self.ensemble_size):
                    filt_models_result_files[j].append(models_result_files[j][i])
        return filt_models_result_files


class SoftmaxMixin:

    def get_ensemble_scores(self, score_func, im_score_file, round_dir, ignore_ims_dict, delete_preds=True):
        f = open(im_score_file, "w")
        train_logits_path = os.path.join(round_dir, "*", self.model_params['train_logits_path'])
        train_results = sorted(list(glob(train_logits_path)))
        im_files = sorted(np.load(train_results[0], mmap_mode='r').files)
        filtered_im_files = [im_file for im_file in im_files if im_file not in ignore_ims_dict[self.im_key]]
        # useful for how to load npz (using "incorrect version): https://stackoverflow.com/questions/61985025/numpy-load-part-of-npz-file-in-mmap-mode
        for im_file in tqdm(filtered_im_files):
            ensemble_preds_arr = []
            for i, result in enumerate(train_results):
                preds_arr = np.load(result, mmap_mode='r')[im_file]
                ensemble_preds_arr.append(preds_arr)
            score = score_func(ensemble_preds_arr)
            f.write(f"{im_file},{np.round(score, 7)}\n")
            f.flush()
        f.close()
        # after obtaining scores, delete the *.npz files for the round
        if delete_preds:
            for result in train_results:
                os.remove(result)
