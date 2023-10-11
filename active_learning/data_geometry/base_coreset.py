import h5py
from glob import glob
from scipy.ndimage.interpolation import zoom
import numpy as np
from tqdm import tqdm
from numpy.random import RandomState
from collections import defaultdict
import os
from active_learning.data_geometry.base_data_geometry import BaseDataGeometry
from active_learning.feature_model.feature_model_factory import FeatureModelFactory
from active_learning.data_geometry import coreset_algs
import json


class BaseCoreset(BaseDataGeometry):
    """Base class for Coreset sampling"""

    def __init__(self, alg_string="kcenter_greedy", metric='euclidean', extra_feature_weight=1.0, phase_weight=1.0,
                 group_weight=1.0, height_weight=1.0, weight_weight=1.0, slice_pos_weight=1.0, patch_size=(256, 256),
                 feature_model=False, feature_model_params=None, contrastive=False, use_model_features=False, seed=0,
                 gpus="cuda:0", **kwargs):
        super().__init__()
        self.alg_string = alg_string
        self.metric = metric
        self.extra_feature_weight = extra_feature_weight
        self.phase_weight = phase_weight
        self.group_weight = group_weight
        self.height_weight = height_weight
        self.weight_weight = weight_weight
        self.slice_pos_weight = slice_pos_weight
        self.patch_size = patch_size
        self.contrastive = contrastive
        self.gpus = gpus
        self.feature_model = feature_model
        self.feature_model_params = feature_model_params
        self.use_model_features = use_model_features
        self.seed = seed
        self.random_state = RandomState(seed=self.seed)
        self.basic_coreset_alg = None
        self.exp_dir = None
        self.data_root = None
        self.all_train_im_files = None
        self.all_train_full_im_paths = None

        self.image_features = None
        self.image_cfgs = None
        self.image_cfgs_arr = None
    
    def setup(self, exp_dir, data_root, all_train_im_files):
        print("Setting up Coreset Class...")
        self.setup_feature_model(exp_dir)
        self.setup_data(data_root, all_train_im_files)
        self.setup_alg()
        print("Done setting up Coreset Class.")

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
        self.setup_image_features()
        print("Done setting up data")

    def setup_image_features(self):
        print("Setting up image features...")
        image_data, self.image_cfgs =  self._get_data(self.all_train_full_im_paths)
        self.image_cfgs_arr = self._process_cfgs()
        self._update_cfgs_arr_indices()
        self.feature_model.init_image_features(image_data)
        print("Done setting up image features")

    def setup_alg(self):
        print("Setting up coreset alg...")
        if self.alg_string in coreset_algs:
            self.coreset_cls = coreset_algs[self.alg_string]
        else:
            raise ValueError(f"No coreset alg found for {self.alg_string}")
        print("Done setting up coreset alg")

    def get_features(self):
        print("Getting features...")
        features = self.feature_model.get_features()
        print("Done getting features")
        return features

    def create_coreset_inst(self, processed_data):
        return self.coreset_cls(processed_data, file_names=self.all_train_im_files, cfgs_arr=self.image_cfgs_arr,
                                metric=self.metric, phase_starting_index=self.phase_starting_index,
                                phase_ending_index=self.phase_ending_index,
                                group_starting_index=self.group_starting_index,
                                group_ending_index=self.group_ending_index,
                                height_starting_index=self.height_starting_index,
                                height_ending_index=self.height_ending_index,
                                weight_starting_index=self.weight_starting_index,
                                weight_ending_index=self.weight_ending_index,
                                slice_rel_pos_starting_index=self.slice_rel_pos_starting_index,
                                slice_rel_pos_ending_index=self.slice_rel_pos_ending_index,
                                slice_pos_starting_index=self.slice_pos_starting_index,
                                slice_pos_ending_index=self.slice_pos_ending_index,
                                extra_feature_weight=self.extra_feature_weight,
                                phase_weight=self.phase_weight, group_weight=self.group_weight,
                                height_weight=self.height_weight, weight_weight=self.weight_weight,
                                slice_pos_weight=self.slice_pos_weight, seed=self.seed)

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
            coreset_inst = self.create_coreset_inst(feat)
        return coreset_inst, feat

    def get_model_features(self, prev_round_dir, train_logits_path, delete_preds=True):
        if prev_round_dir is None:
            print("No model features for first round!")
            return None
        train_logits_path = os.path.join(prev_round_dir, "*", train_logits_path)
        print(f"Looking for model features in {train_logits_path}")
        train_results = sorted(list(glob(train_logits_path)))
        if len(train_results) == 1:
            train_result = train_results[0]
            print(f"Found {train_result}")
        elif len(train_results) > 1:
            best_perf = 0
            best_index = None
            for index, train_result in enumerate(train_results):
                model_dir = os.path.dirname(train_result)
                val_file = os.path.join(model_dir, "val_metrics.json")
                with open(val_file, "r") as json_file:
                    val_metrics = json.load(json_file)
                val_result = val_metrics["performance"]
                if val_result> best_perf:
                    best_perf = val_result
                    best_index = index
            train_result = train_results[best_index]
            print(f"Found best {train_result}")
        else:
            raise ValueError("No model features found!")

        # useful for how to load npz (using "incorrect version): https://stackoverflow.com/questions/61985025/numpy-load-part-of-npz-file-in-mmap-mode
        preds_arrs = []
        for im_file in tqdm(self.all_train_im_files):
            preds_arr = np.load(train_result, mmap_mode='r')[os.path.basename(im_file)]
            preds_arrs.append(preds_arr)
        preds_arrs = np.stack(preds_arrs, axis=0)
        # flatten array except for first dim
        preds_arrs = preds_arrs.reshape(preds_arrs.shape[0], -1)

        # after obtaining features, delete the *.npz files for the round
        if delete_preds:
            for train_result_ in train_results:
                os.remove(train_result_)

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

    def _load_cfg(self, cfg_file):
        cfg = {}
        with open(cfg_file, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                if key == 'ED' or key == 'ES':
                    value = int(value)
                if key == 'Height' or key == 'Weight':
                    value = float(value)
                cfg[key] = value
        return cfg

    def _get_data(self, all_train_full_im_paths):
        cases = []
        cfgs = []
        for im_path in tqdm(all_train_full_im_paths):
            image = self._load_image(im_path)
            cfg_path = self._get_cfg_path(im_path)
            cfg = self._load_cfg(cfg_path)
            cases.append(image)
            cfgs.append(cfg)
        cases_arr = np.concatenate(cases, axis=0)
        return cases_arr, cfgs

    def _get_cfg_path(self, im_path):
        cfg_path = self._extract_patient_prefix(im_path) + ".cfg"
        return cfg_path

    def _process_cfgs(self):
        # calculate number of slices per frame
        num_slices_dict = dict()
        for file_name in self.all_train_im_files:
            patient_frame_no = self._extract_patient_frame_no_str(file_name)
            if patient_frame_no not in num_slices_dict:
                num_slices_dict[patient_frame_no] = 1
            else:
                num_slices_dict[patient_frame_no] += 1

        # calculate number of groups
        groups_dict = defaultdict(lambda: len(groups_dict))
        for im_cfg in self.image_cfgs:
            groups_dict[im_cfg['Group']]
        self.num_groups = len(groups_dict)
        one_hot_group = [0] * self.num_groups

        # calculate z-score params for height and weight
        height_mean = 0
        height_sstd = 0
        weight_mean = 0
        weight_sstd = 0
        for im_cfg in self.image_cfgs:
            height_mean += im_cfg['Height']
            weight_mean += im_cfg['Weight']
        height_mean /= len(self.image_cfgs)
        weight_mean /= len(self.image_cfgs)

        for im_cfg in self.image_cfgs:
            height_sstd += (im_cfg['Height'] - height_mean) ** 2
            weight_sstd += (im_cfg['Weight'] - weight_mean) ** 2
        height_sstd = (height_sstd / (len(self.image_cfgs) - 1)) ** (.5)
        weight_sstd = (weight_sstd / (len(self.image_cfgs) - 1)) ** (.5)

        # encode all cfg features
        extra_features_lst = []
        for im_cfg, file_name in zip(self.image_cfgs, self.all_train_im_files):
            extra_features = []
            # add if frame is ED or ES (one hot encoded)
            patient_frame_no = self._extract_patient_frame_no_str(file_name)
            frame_num = self._extract_frame_no(file_name)
            if im_cfg['ED'] == frame_num:
                extra_features.append(1)
            elif im_cfg['ES'] == frame_num:
                extra_features.append(0)
            else:
                raise ValueError("Frame number not found in ED or ES")
            # add Group ID (one hot encoded)
            group_id = groups_dict[im_cfg['Group']]
            im_one_hot_group = one_hot_group.copy()
            im_one_hot_group[group_id] = 1
            extra_features.extend(im_one_hot_group)

            # add Height and Weight
            z_score_height = (im_cfg['Height'] - height_mean) / height_sstd
            z_score_weight = (im_cfg['Weight'] - weight_mean) / weight_sstd

            extra_features.append(z_score_height)
            extra_features.append(z_score_weight)

            # add relative slice position and general slice position
            slice_num = self._extract_slice_no(file_name)
            extra_features.append(slice_num / num_slices_dict[patient_frame_no])
            extra_features.append(slice_num)

            extra_features_lst.append(np.array(extra_features))

        return np.array(extra_features_lst)

    def _update_cfgs_arr_indices(self):
        self.phase_starting_index = 0
        self.phase_ending_index = self.phase_starting_index + 1
        self.group_starting_index = self.phase_ending_index
        self.group_ending_index = self.group_starting_index + self.num_groups
        self.height_starting_index = self.group_ending_index
        self.height_ending_index = self.height_starting_index + 1
        self.weight_starting_index = self.height_ending_index
        self.weight_ending_index = self.weight_starting_index + 1
        self.slice_rel_pos_starting_index = self.weight_ending_index
        self.slice_rel_pos_ending_index = self.slice_rel_pos_starting_index + 1
        self.slice_pos_starting_index = self.slice_rel_pos_ending_index
        self.slice_pos_ending_index = self.slice_pos_starting_index + 1

    def _extract_patient_frame_no_str(self, im_path):
        return self._extract_patient_prefix(im_path) + "_" + str(self._extract_frame_no(im_path))

    def _extract_patient_prefix(self, im_path):
        patient_prefix_end_index = self._get_patient_prefix_end_index(im_path)
        patient_prefix = im_path[:patient_prefix_end_index]
        return patient_prefix

    def _get_patient_prefix_end_index(self, im_path):
        patient_prefix = "patient"
        patient_prefix_len = len(patient_prefix)
        patient_prefix_index = im_path.index(patient_prefix)
        patient_num_len = 3
        patient_prefix_end_index = patient_prefix_index + patient_prefix_len + patient_num_len
        return patient_prefix_end_index

    def _extract_frame_no(self, im_path):
        frame_prefix = "frame"
        frame_num_len = 2
        frame_and_num_prefix_len = len(frame_prefix) + frame_num_len
        frame_and_num_end_index = im_path.index(frame_prefix) + frame_and_num_prefix_len
        frame_no = im_path[frame_and_num_end_index-frame_num_len:frame_and_num_end_index]
        return int(frame_no)

    def _extract_slice_no(self, im_path):
        slice_str = "slice_"
        slice_str_len = len(slice_str)
        slice_num_len = 1
        slice_start_index = im_path.index(slice_str) + slice_str_len
        slice_end_index = slice_start_index + slice_num_len
        slice_num = int(im_path[slice_start_index:slice_end_index])
        return slice_num