import h5py
from glob import glob
from scipy.ndimage.interpolation import zoom
import numpy as np
from tqdm import tqdm
from numpy.random import RandomState
from collections import defaultdict
from functools import partial
import os
from active_learning.data_geometry.base_data_geometry import BaseDataGeometry
from active_learning.feature_model.feature_model_factory import FeatureModelFactory
from active_learning.data_geometry import coreset_algs
import json
from active_learning.data_geometry.dist_metrics import metric_w_config


class BaseCoreset(BaseDataGeometry):
    """Base class for Coreset sampling"""

    def __init__(self, alg_string="kcenter_greedy", metric='euclidean', coreset_kwargs=None, use_uncertainty=False,
                 model_uncertainty=None, uncertainty_score_file="entropy.txt", use_labels=False, ann_type=None, label_wt=1.0,
                 max_dist=None, wt_max_dist_mult=1.0, extra_feature_wt=0.0, patient_wt=0.0, phase_wt=0.0, group_wt=0.0,
                 height_wt=0.0, weight_wt=0.0, slice_rel_pos_wt=0.0, slice_mid_wt=0.0, slice_pos_wt=0.0,
                 uncertainty_wt=0.0, patch_size=(256, 256), feature_model=False, feature_model_params=None,
                 contrastive=False, use_model_features=False, seed=0, gpus="cuda:0", **kwargs):
        super().__init__()
        self.alg_string = alg_string
        self.metric = metric
        if coreset_kwargs is None:
            self.coreset_kwargs = {}
        elif isinstance(coreset_kwargs, dict):
            self.coreset_kwargs = coreset_kwargs
        else:
            raise ValueError("coreset_kwargs must be a dict or None")
        self.use_uncertainty = use_uncertainty
        self.model_uncertainty = model_uncertainty
        self.uncertainty_score_file = uncertainty_score_file
        self.ann_type = ann_type
        self.use_labels = use_labels
        self.label_wt = label_wt
        self.max_dist = max_dist
        self.wt_max_dist_mult = wt_max_dist_mult

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
        self.image_meta_data = None
        self.image_meta_data_arr = None
        self.non_image_wts = None
        self.non_image_indices = None
        self._update_non_image_wts(extra_feature_wt=extra_feature_wt, patient_wt=patient_wt, phase_wt=phase_wt,
                                   group_wt=group_wt, height_wt=height_wt, weight_wt=weight_wt,
                                   slice_rel_pos_wt=slice_rel_pos_wt, slice_mid_wt=slice_mid_wt, slice_pos_wt=slice_pos_wt,
                                   uncertainty_wt=uncertainty_wt, wt_max_dist_mult=wt_max_dist_mult)
    
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
        print("Getting data")
        image_data, self.image_labels_arr, self.image_meta_data =  self._get_data(self.all_train_full_im_paths)
        print("Processing meta_data...")
        self.image_meta_data_arr = self._process_meta_data()
        self._update_non_image_indices()
        print("Initializing image features for feature model...")
        self.feature_model.init_image_features(image_data, self.image_meta_data_arr, self.non_image_indices)
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
        # assert len(features.shape) == 2
        print("Done getting features")
        return features

    def create_coreset_inst(self, processed_data, prev_round_dir=None, uncertainty_kwargs=None):
        print("Creating coreset instance...")
        print("Getting coreset metric and features...")
        coreset_metric, features = self.get_coreset_metric_and_features(processed_data, meta_data_arr=self.image_meta_data_arr,
                                                                        prev_round_dir=prev_round_dir,
                                                                        uncertainty_kwargs=uncertainty_kwargs)
        coreset_inst = self.coreset_cls(X=features, file_names=self.all_train_im_files, metric=coreset_metric,
                                        **self.coreset_kwargs)
        print(f"Created {coreset_inst.name} inst!")
        return coreset_inst


    def get_coreset_inst_and_features_for_round(self, prev_round_dir, train_logits_path, uncertainty_kwargs=None,
                                                delete_preds=True):
        if self.use_model_features:
            print("Using Model Features")
            feat = self.get_model_features(prev_round_dir, train_logits_path, delete_preds=delete_preds)
            # assert len(feat.shape) == 2
            if feat is None:
                print("Model features not found. Using image features instead")
                feat = self.get_features()
        else:
            feat = self.get_features()
        coreset_inst = self.create_coreset_inst(feat, prev_round_dir=prev_round_dir,
                                                uncertainty_kwargs=uncertainty_kwargs)
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

    def get_coreset_metric_and_features(self, processed_data, meta_data_arr, prev_round_dir=None, uncertainty_kwargs=None):
        num_im_features = processed_data.shape[1]
        if self.use_labels:
            print(f"Using labels with weight {self.label_wt}")
            labels = self.image_labels_arr.reshape(processed_data.shape[0], -1)
            # check that number of pixels in image is greater than number of labels
            # only update points that are part of the image features
            assert processed_data.shape[1] >= labels.shape[1]
            im_features = processed_data[:, :labels.shape[1]]
            label_mask = np.where(labels != 4, self.label_wt, 1)
            im_features = im_features * label_mask
            processed_data[:, :labels.shape[1]] = im_features
        features = np.concatenate([processed_data, meta_data_arr], axis=1)
        if self.use_uncertainty and (prev_round_dir is not None):
            if uncertainty_kwargs is None:
                uncertainty_kwargs = dict()
            uncertainty_round_score_file = os.path.join(prev_round_dir, self.uncertainty_score_file)
            self.model_uncertainty.calculate_uncertainty(im_score_file=uncertainty_round_score_file,
                                                         round_dir=prev_round_dir, **uncertainty_kwargs)
            uncertainty_features = self._extract_uncertainty_features(uncertainty_round_score_file)
            assert uncertainty_features.shape[0] == features.shape[0]
            features = np.concatenate([features, uncertainty_features], axis=1)
        elif self.use_uncertainty:
            print("No uncertainty features for first round!")
        non_image_kwargs = {}
        if self.non_image_indices is not None:
            non_image_kwargs.update(self.non_image_indices)
        if self.non_image_wts is not None:
            non_image_kwargs.update(self.non_image_wts)
        coreset_metric = partial(metric_w_config, image_metric=self.metric, max_dist=self.max_dist, num_im_features=num_im_features,
                                 **non_image_kwargs)
        return coreset_metric, features

    @staticmethod
    def _patch_im(im, patch_size):
        x, y = im.shape
        image = zoom(im, (patch_size[0] / x, patch_size[1] / y), order=0)
        return image

    def _load_image_and_label(self, case):
        h5f = h5py.File(case, 'r')
        image = h5f['image'][:]
        patched_image = self._patch_im(image, self.patch_size)
        patched_image = patched_image[np.newaxis,]
        if self.use_labels:
            label = h5f[self.ann_type][:]
            patched_label = self._patch_im(label, self.patch_size)
            return patched_image, patched_label[np.newaxis,]
        else:
            return patched_image, None

    def _load_meta_data(self, meta_data_file):
        meta_data = {}
        with open(meta_data_file, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                if key == 'ED' or key == 'ES':
                    value = int(value)
                if key == 'Height' or key == 'Weight':
                    value = float(value)
                meta_data[key] = value
        return meta_data

    def _get_data(self, all_train_full_im_paths):
        if self.use_labels:
            cases_arr, labels_arr, meta_data = self._get_image_and_label_data(all_train_full_im_paths)
        else:
            labels_arr = None
            cases_arr, meta_data = self._get_image_data(all_train_full_im_paths)
        return cases_arr, labels_arr, meta_data

    def _get_image_data(self, all_train_full_im_paths):
        cases = []
        meta_data = []
        for im_path in tqdm(all_train_full_im_paths):
            image, _ = self._load_image_and_label(im_path)
            meta_data_path = self._get_meta_data_path(im_path)
            meta_datum = self._load_meta_data(meta_data_path)
            cases.append(image)
            meta_data.append(meta_datum)
        cases_arr = np.concatenate(cases, axis=0)
        return cases_arr, meta_data

    def _get_image_and_label_data(self, all_train_full_im_paths):
        cases = []
        labels = []
        meta_data = []
        for im_path in tqdm(all_train_full_im_paths):
            image, label = self._load_image_and_label(im_path)
            meta_data_path = self._get_meta_data_path(im_path)
            meta_datum = self._load_meta_data(meta_data_path)
            cases.append(image)
            labels.append(label)
            meta_data.append(meta_datum)
        cases_arr = np.concatenate(cases, axis=0)
        labels_arr = np.concatenate(labels, axis=0)
        return cases_arr, labels_arr, meta_data

    def _get_meta_data_path(self, im_path):
        meta_data_path = self._extract_patient_prefix(im_path) + ".cfg"
        return meta_data_path

    def _process_meta_data(self):
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
        for im_meta_datum in self.image_meta_data:
            groups_dict[im_meta_datum['Group']]
        self.num_groups = len(groups_dict)
        one_hot_group = [0] * self.num_groups

        # calculate z-score params for height and weight
        height_mean = 0
        height_sstd = 0
        weight_mean = 0
        weight_sstd = 0
        for im_meta_datum in self.image_meta_data:
            height_mean += im_meta_datum['Height']
            weight_mean += im_meta_datum['Weight']
        height_mean /= len(self.image_meta_data)
        weight_mean /= len(self.image_meta_data)

        for im_meta_datum in self.image_meta_data:
            height_sstd += (im_meta_datum['Height'] - height_mean) ** 2
            weight_sstd += (im_meta_datum['Weight'] - weight_mean) ** 2
        height_sstd = (height_sstd / (len(self.image_meta_data) - 1)) ** (.5)
        weight_sstd = (weight_sstd / (len(self.image_meta_data) - 1)) ** (.5)

        # encode all cfg features
        extra_features_lst = []
        for im_meta_datum, file_name in zip(self.image_meta_data, self.all_train_im_files):
            extra_features = []
            # add patient number
            patient_num = self._extract_patient_num(file_name)
            extra_features.append(patient_num)
            # add if frame is ED or ES (one hot encoded)
            patient_frame_no = self._extract_patient_frame_no_str(file_name)
            frame_num = self._extract_frame_no(file_name)
            if im_meta_datum['ED'] == frame_num:
                extra_features.append(1)
            elif im_meta_datum['ES'] == frame_num:
                extra_features.append(0)
            else:
                raise ValueError("Frame number not found in ED or ES")
            # add Group ID (one hot encoded)
            group_id = groups_dict[im_meta_datum['Group']]
            im_one_hot_group = one_hot_group.copy()
            im_one_hot_group[group_id] = 1
            extra_features.extend(im_one_hot_group)

            # add Height and Weight
            z_score_height = (im_meta_datum['Height'] - height_mean) / height_sstd
            z_score_weight = (im_meta_datum['Weight'] - weight_mean) / weight_sstd

            extra_features.append(z_score_height)
            extra_features.append(z_score_weight)

            # add relative slice position and general slice position
            slice_num = self._extract_slice_no(file_name)
            extra_features.append(slice_num / num_slices_dict[patient_frame_no])
            extra_features.append(slice_num)

            extra_features_lst.append(np.array(extra_features))

        return np.array(extra_features_lst)

    def _update_non_image_wts(self, extra_feature_wt=None, patient_wt=None, phase_wt=None, group_wt=None, height_wt=None,
                              weight_wt=None, slice_rel_pos_wt=None, slice_mid_wt=None, slice_pos_wt=None, uncertainty_wt=None,
                              wt_max_dist_mult=None):
        self.non_image_wts = {'extra_feature_wt': extra_feature_wt, 'patient_wt': patient_wt, 'phase_wt': phase_wt,
                               'group_wt': group_wt, 'height_wt': height_wt, 'weight_wt': weight_wt,
                               'slice_rel_pos_wt': slice_rel_pos_wt, 'slice_mid_wt': slice_mid_wt,
                               'slice_pos_wt': slice_pos_wt, "uncertainty_wt": uncertainty_wt,
                               "wt_max_dist_mult": wt_max_dist_mult}

    def _update_non_image_indices(self):
        self.non_image_indices = dict()
        self.non_image_indices['patient_starting_index'] = 0
        self.non_image_indices['patient_ending_index'] = self.non_image_indices['patient_starting_index'] + 1
        self.non_image_indices['phase_starting_index'] = self.non_image_indices['patient_ending_index']
        self.non_image_indices['phase_ending_index'] = self.non_image_indices['phase_starting_index'] + 1
        self.non_image_indices['group_starting_index'] = self.non_image_indices['phase_ending_index']
        self.non_image_indices['group_ending_index'] = self.non_image_indices['group_starting_index'] + self.num_groups
        self.non_image_indices['height_starting_index'] = self.non_image_indices['group_ending_index']
        self.non_image_indices['height_ending_index'] = self.non_image_indices['height_starting_index'] + 1
        self.non_image_indices['weight_starting_index'] = self.non_image_indices['height_ending_index']
        self.non_image_indices['weight_ending_index'] = self.non_image_indices['weight_starting_index'] + 1
        self.non_image_indices['slice_rel_pos_starting_index'] = self.non_image_indices['weight_ending_index']
        self.non_image_indices['slice_rel_pos_ending_index'] = self.non_image_indices['slice_rel_pos_starting_index'] + 1
        self.non_image_indices['slice_pos_starting_index'] = self.non_image_indices['slice_rel_pos_ending_index']
        self.non_image_indices['slice_pos_ending_index'] = self.non_image_indices['slice_pos_starting_index'] + 1
        self.non_image_indices['uncertainty_starting_index'] = self.non_image_indices['slice_pos_ending_index']
        self.non_image_indices['uncertainty_ending_index'] = self.non_image_indices['uncertainty_starting_index'] + 1

    def _extract_uncertainty_features(self, score_file):
        # get the scores per image from the score file in this format: img_name, score
        im_scores_list = open(score_file).readlines()
        im_scores_list = [im_score.strip().split(",") for im_score in im_scores_list]
        im_scores_list = [(im_score[0], float(im_score[1])) for im_score in im_scores_list]
        sorted_im_scores_list = sorted(im_scores_list, key=lambda x: x[0])
        sorted_score_list = [im_score[1] for im_score in sorted_im_scores_list]

        return np.array(sorted_score_list).reshape(-1, 1)


    def _extract_patient_frame_no_str(self, im_path):
        return self._extract_patient_prefix(im_path) + "_" + str(self._extract_frame_no(im_path))

    def _get_patient_num_start_index(self, im_path):
        patient_prefix = "patient"
        patient_prefix_len = len(patient_prefix)
        patient_prefix_index = im_path.index(patient_prefix)
        patient_num_start_index = patient_prefix_index + patient_prefix_len
        return patient_num_start_index

    def _get_patient_num_end_index(self, im_path):
        patient_num_len = 3
        patient_num_start_index = self._get_patient_num_start_index(im_path)
        return patient_num_start_index + patient_num_len

    def _extract_patient_num(self, im_path):
        return int(im_path[self._get_patient_num_start_index(im_path):self._get_patient_num_end_index(im_path)])

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