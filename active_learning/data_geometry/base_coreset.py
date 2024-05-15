from glob import glob
import torch
import numpy as np
from tqdm import tqdm
from numpy.random import RandomState
from functools import partial
import os
from active_learning.data_geometry.base_data_geometry import BaseDataGeometry
from active_learning.feature_model.feature_model_factory import FeatureModelFactory
from active_learning.dataset.dataset_factory import DatasetFactory
from active_learning.data_geometry import coreset_algs
import json
from active_learning.data_geometry.dist_metrics import metric_w_config


class BaseCoreset(BaseDataGeometry):
    """Base class for Coreset sampling"""

    def __init__(self, alg_string="kcenter_greedy", metric='euclidean', coreset_kwargs=None, dataset_type="ACDC",
                 in_chns=1, dataset_kwargs=None, use_uncertainty=False, model_uncertainty=None,
                 uncertainty_score_file="entropy.txt", use_labels=False, ann_type=None, label_wt=1.0, max_dist=None,
                 wt_max_dist_mult=1.0, extra_feature_wt=0.0, patient_wt=0.0, phase_wt=0.0, group_wt=0.0, height_wt=0.0,
                 weight_wt=0.0, slice_rel_pos_wt=0.0, slice_mid_wt=0.0, slice_pos_wt=0.0, uncertainty_wt=0.0,
                 feature_model=False, feature_model_params=None, contrastive=False, use_model_features=False,
                 analyze_dataset=False, seed=0, gpus="cuda:0", **kwargs):
        super().__init__()
        self.alg_string = alg_string
        self.metric = metric
        if coreset_kwargs is None:
            self.coreset_kwargs = {}
        elif isinstance(coreset_kwargs, dict):
            self.coreset_kwargs = coreset_kwargs
        else:
            raise ValueError("coreset_kwargs must be a dict or None")
        self.dataset_type = dataset_type
        if not isinstance(dataset_kwargs, dict):
            self.dataset_kwargs = {}
        else:
            self.dataset_kwargs = dataset_kwargs
        if not isinstance(feature_model_params, dict):
            self.feature_model_params = {}
        else:
            self.feature_model_params = feature_model_params
        self.in_chns = in_chns
        self.use_uncertainty = use_uncertainty
        self.model_uncertainty = model_uncertainty
        self.uncertainty_score_file = uncertainty_score_file
        self.ann_type = ann_type
        self.use_labels = use_labels
        self.label_wt = label_wt
        self.max_dist = max_dist
        self.wt_max_dist_mult = wt_max_dist_mult
        self.contrastive = contrastive
        self.gpus = gpus
        self.feature_model = feature_model
        self.fuse_image_data = self.feature_model_params.get("fuse_image_data", False)
        self.use_model_features = use_model_features
        self.analyze_dataset = analyze_dataset
        self.seed = seed
        self.random_state = RandomState(seed=self.seed)
        self.basic_coreset_alg = None
        self.exp_dir = None
        self.data_root = None
        self.all_train_im_files = None
        self.all_train_full_im_paths = None
        self.dataset = None
        self.image_features = None
        self.image_meta_data = None
        self.image_meta_data_arr = None
        self.non_image_wts = None
        self.non_image_indices = None
        self._update_non_image_wts(extra_feature_wt=extra_feature_wt, patient_wt=patient_wt, phase_wt=phase_wt,
                                   group_wt=group_wt, height_wt=height_wt, weight_wt=weight_wt,
                                   slice_rel_pos_wt=slice_rel_pos_wt, slice_mid_wt=slice_mid_wt,
                                   slice_pos_wt=slice_pos_wt,
                                   uncertainty_wt=uncertainty_wt, wt_max_dist_mult=wt_max_dist_mult)

    def setup(self, exp_dir, data_root, all_train_im_files):
        print("Setting up Coreset Class...")
        self.setup_feature_model(exp_dir)
        self.setup_data(data_root, all_train_im_files)
        self.setup_alg()
        print("Done setting up Coreset Class.")
        if self.analyze_dataset:
            self.print_dataset_analysis()

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
        print("Done setting up data")

    def setup_image_features(self):
        print("Setting up image features...")
        print("Getting data")
        image_data, self.image_meta_data, self.image_labels_arr = self.dataset.get_data()
        print("Processing meta_data...")
        self.image_meta_data_arr = self.dataset.process_meta_data(self.image_meta_data)
        self.non_image_indices = self.dataset.get_non_image_indices()
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
        coreset_metric, features = self.get_coreset_metric_and_features(processed_data, prev_round_dir=prev_round_dir,
                                                                        uncertainty_kwargs=uncertainty_kwargs)
        coreset_inst = self.coreset_cls(X=features, file_names=self.all_train_im_files, metric=coreset_metric,
                                        num_im_features=processed_data.shape[1],
                                        uncertainty_starting_index=self.non_image_indices[
                                            "uncertainty_starting_index"] if self.use_uncertainty else None,
                                        uncertainty_ending_index=self.non_image_indices[
                                            "uncertainty_ending_index"] if self.use_uncertainty else None,
                                        uncertainty_wt=self.non_image_wts[
                                            "uncertainty_wt"] if self.use_uncertainty else None,
                                        gpus=self.gpus, **self.coreset_kwargs)
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
                if val_result > best_perf:
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

    def get_coreset_metric_and_features(self, processed_data, prev_round_dir=None, uncertainty_kwargs=None):
        num_im_features = processed_data.shape[1]
        # check that number of pixels in image is greater than number of labels
        # only update points that are part of the image features
        if self.use_labels:
            print(f"Using labels with weight {self.label_wt}")
            labels = self.image_labels_arr.reshape(processed_data.shape[0], -1)
            assert processed_data.shape[1] >= labels.shape[1]
            num_label_features = labels.shape[1]
            # hard-coded number of classes to be (background + 3)
            label_mask = np.where(labels < 4, self.label_wt, 1)
            features = np.concatenate([processed_data, label_mask, self.image_meta_data_arr], axis=1)
        # hard code 1 labels for hacky fix to the metric to keep track of position
        elif self.fuse_image_data:
            labels = self.image_labels_arr.reshape(processed_data.shape[0], -1)
            assert processed_data.shape[1] >= labels.shape[1]
            num_label_features = labels.shape[1]
            label_mask = np.ones(labels.shape)
            features = np.concatenate([processed_data, label_mask, self.image_meta_data_arr], axis=1)
        else:
            num_label_features = 0
            features = np.concatenate([processed_data, self.image_meta_data_arr], axis=1)
        if self.use_uncertainty and (prev_round_dir is not None):
            if uncertainty_kwargs is None:
                uncertainty_kwargs = dict()
            uncertainty_round_score_file = os.path.join(prev_round_dir, self.uncertainty_score_file)
            if not os.path.exists(uncertainty_round_score_file):
                self.model_uncertainty.model.train_ensemble(round_dir=prev_round_dir,
                                                            cur_total_oracle_split=None,
                                                            cur_total_pseudo_split=None,
                                                            train=False, inf_train=True,
                                                            inf_val=False, inf_test=False)
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
        coreset_metric = partial(metric_w_config, image_metric=self.metric, max_dist=self.max_dist,
                                 num_im_features=num_im_features,**non_image_kwargs)
        return coreset_metric, features

    def _update_non_image_wts(self, extra_feature_wt=None, patient_wt=None, phase_wt=None, group_wt=None,
                              height_wt=None,
                              weight_wt=None, slice_rel_pos_wt=None, slice_mid_wt=None, slice_pos_wt=None,
                              uncertainty_wt=None,
                              wt_max_dist_mult=None):
        self.non_image_wts = {'extra_feature_wt': extra_feature_wt, 'patient_wt': patient_wt, 'phase_wt': phase_wt,
                              'group_wt': group_wt, 'height_wt': height_wt, 'weight_wt': weight_wt,
                              'slice_rel_pos_wt': slice_rel_pos_wt, 'slice_mid_wt': slice_mid_wt,
                              'slice_pos_wt': slice_pos_wt, "uncertainty_wt": uncertainty_wt,
                              "wt_max_dist_mult": wt_max_dist_mult}

    def _extract_uncertainty_features(self, score_file):
        # get the scores per image from the score file in this format: img_name, score
        im_scores_list = open(score_file).readlines()
        im_scores_list = [im_score.strip().split(",") for im_score in im_scores_list]
        im_scores_list = [(im_score[0], float(im_score[1])) for im_score in im_scores_list]
        sorted_im_scores_list = sorted(im_scores_list, key=lambda x: x[0])
        sorted_score_list = [im_score[1] for im_score in sorted_im_scores_list]

        return np.array(sorted_score_list).reshape(-1, 1)

    # TODO: Generalize for other datasets; hard-coded for ACDC (just need to use config parameters)
    def print_dataset_analysis(self):
        flat_image_data = self.feature_model.flat_image_data
        mean_image_data = np.mean(flat_image_data, axis=0)
        print(f"min: {np.min(flat_image_data)}, max: {np.max(flat_image_data)}, mean: {np.mean(flat_image_data)}, std: {np.std(flat_image_data)}")
        """
        # calculate mean absolute deviation (overall)
        mad = np.mean(np.abs(flat_image_data - mean_image_data))
        stad = np.std(np.abs(flat_image_data - mean_image_data))
        print(f"MAD (overall): {mad}, STAD (overall): {stad}")
        # calculate mean absolute deviation (patient)
        patient_ids = np.unique(self.image_meta_data_arr[:, 0])
        volume_ids = np.unique(self.image_meta_data_arr[:, 1])
        slice_pos_lst = np.unique(self.image_meta_data_arr[:, -1])
        patient_ads = []
        for patient_id in patient_ids:
            patient_indices = np.where(self.image_meta_data_arr[:, 0] == patient_id)[0]
            patient_mean_image_data = np.mean(flat_image_data[patient_indices], axis=0)
            patient_ads.extend(np.abs(flat_image_data[patient_indices] - patient_mean_image_data))
        patient_mad = np.mean(patient_ads)
        patient_stads = np.std(patient_ads)
        print(f"MAD (patient): {patient_mad}, STAD (patient): {patient_stads}")
        # calculate mean absolute deviation (volume)
        volume_ads = []
        for patient_id in patient_ids:
            for volume_id in volume_ids:
                volume_indices = np.where(
                    (self.image_meta_data_arr[:, 0] == patient_id) & (self.image_meta_data_arr[:, 1] == volume_id))[0]
                volume_mean_image_data = np.mean(flat_image_data[volume_indices], axis=0)
                volume_ads.extend(np.abs(flat_image_data[volume_indices] - volume_mean_image_data))
        volume_mad = np.mean(volume_ads)
        volume_stads = np.std(volume_ads)
        print(f"MAD (volume): {volume_mad}, STAD (volume): {volume_stads}")
        # calculate mean absolute deviation (volume)
        other_volume_ads = []
        for patient_id in patient_ids:
            for volume_id in volume_ids:
                volume_indices = np.where(
                    (self.image_meta_data_arr[:, 0] == patient_id) & (self.image_meta_data_arr[:, 1] == volume_id))[0]
                for volume_index in volume_indices:
                    other_volume_indices = [i for i in volume_indices if i != volume_index]
                    other_volume_mean_image_data = np.mean(flat_image_data[other_volume_indices], axis=0)
                    other_volume_ads.extend(np.abs(flat_image_data[volume_index] - other_volume_mean_image_data))
        other_volume_mad = np.mean(other_volume_ads)
        other_volume_stads = np.std(other_volume_ads)
        print(f"MAD (other-volume): {other_volume_mad}, STAD (other-volume): {other_volume_stads}")
        # calculate mean absolute deviation (patientnon-volume)
        nonvolume_ads = []
        for patient_id in patient_ids:
            for volume_id in volume_ids:
                volume_indices = np.where(
                    (self.image_meta_data_arr[:, 0] == patient_id) & (self.image_meta_data_arr[:, 1] == volume_id))[0]
                nonvolume_indices = np.where(
                    (self.image_meta_data_arr[:, 0] == patient_id) & (self.image_meta_data_arr[:, 1] != volume_id))[0]
                nonvolume_mean_image_data = np.mean(flat_image_data[nonvolume_indices], axis=0)
                nonvolume_ads.extend(np.abs(flat_image_data[volume_indices] - nonvolume_mean_image_data))
        nonvolume_mad = np.mean(nonvolume_ads)
        nonvolume_stads = np.std(nonvolume_ads)
        print(f"MAD (non-volume): {nonvolume_mad}, STAD (non-volume): {nonvolume_stads}")
        # calculate mean absolute deviation (slice-adjacency)
        slice_ads = []
        for patient_id in patient_ids:
            for volume_id in volume_ids:
                for slice_pos in slice_pos_lst:
                    slice_index = np.where((self.image_meta_data_arr[:, 0] == patient_id) &
                                           ( self.image_meta_data_arr[:, 1] == volume_id) &
                                           (self.image_meta_data_arr[:, -1] == slice_pos))[0]
                    slice_prev_index = np.where((self.image_meta_data_arr[:, 0] == patient_id) &
                                                ( self.image_meta_data_arr[:, 1] == volume_id) &
                                                (self.image_meta_data_arr[:, -1] == (slice_pos - 1)))[0]
                    slice_next_index = np.where((self.image_meta_data_arr[:, 0] == patient_id) &
                                                (self.image_meta_data_arr[:, 1] == volume_id) &
                                                (self.image_meta_data_arr[:, -1] == (slice_pos + 1)))[0]
                    if len(slice_index) == 0:
                        continue
                    assert (len(slice_prev_index) != 0) or (
                                len(slice_next_index) != 0), "both previous and next slice are missing"
                    if len(slice_prev_index) == 0:
                        slice_prev_index = slice_next_index
                    if len(slice_next_index) == 0:
                        slice_next_index = slice_prev_index
                    slice_mean_image_data = (flat_image_data[slice_index] + flat_image_data[slice_prev_index] +
                                             flat_image_data[slice_next_index]) / 3
                    slice_ads.extend(np.abs(flat_image_data[slice_index] - slice_mean_image_data))
        slice_mad = np.mean(slice_ads)
        slice_stad = np.std(slice_ads)
        print(f"MAD (slice-adjacency): {slice_mad}, STAD (slice-adjacency): {slice_stad}")
        """
        patient_ids = np.unique(self.image_meta_data_arr[:, 0])
        volume_ids = np.unique(self.image_meta_data_arr[:, 1])
        slice_pos_lst = np.unique(self.image_meta_data_arr[:, -1])

        flat_image_data_tensor = torch.tensor(flat_image_data, dtype=torch.float32).cuda()

        # calculate pairwise differences (overall)
        ads = self.calculate_abs_pairwise_diff(flat_image_data_tensor)
        mpad = torch.mean(ads)
        stpad = torch.std(ads)
        print(f"MPAD: {mpad}, STPAD: {stpad}")

        # calculate pairwise differences (patient)
        patient_ads = []
        for patient_id in patient_ids:
            patient_indices = torch.where(self.image_meta_data_arr[:, 0] == patient_id)[0]
            patient_flat_image_data = flat_image_data[patient_indices]
            ads = self.calculate_abs_pairwise_diff(patient_flat_image_data)
            patient_ads.extend(ads)
        patient_mpad = torch.mean(patient_ads)
        patient_stpad = torch.std(patient_ads)
        print(f"Patient MPAD: {patient_mpad}, Patient STPAD: {patient_stpad}")

        # calculate pairwise differences (volume)
        volume_ads = []
        for patient_id in patient_ids:
            for volume_id in volume_ids:
                volume_indices = torch.where(
                    (self.image_meta_data_arr[:, 0] == patient_id) & (self.image_meta_data_arr[:, 1] == volume_id))[0]
                volume_flat_image_data = flat_image_data[volume_indices]
                ads = self.calculate_abs_pairwise_diff(volume_flat_image_data)
                volume_ads.extend(ads)
        volume_mpad = torch.mean(patient_ads)
        volume_stpad = torch.std(patient_ads)
        print(f"Volume MPAD: {volume_mpad}, Volume STPAD: {volume_stpad}")

        # calculate pairwise differences (slice-adjacency)
        """
        for patient_id in patient_ids:
            for volume_id in volume_ids:
                for slice_pos in slice_pos_lst:
                    slice_index = np.where((self.image_meta_data_arr[:, 0] == patient_id) &
                                           ( self.image_meta_data_arr[:, 1] == volume_id) &
                                           (self.image_meta_data_arr[:, -1] == slice_pos))[0]
                    slice_prev_index = np.where((self.image_meta_data_arr[:, 0] == patient_id) &
                                                ( self.image_meta_data_arr[:, 1] == volume_id) &
                                                (self.image_meta_data_arr[:, -1] == (slice_pos - 1)))[0]
                    slice_next_index = np.where((self.image_meta_data_arr[:, 0] == patient_id) &
                                                (self.image_meta_data_arr[:, 1] == volume_id) &
                                                (self.image_meta_data_arr[:, -1] == (slice_pos + 1)))[0]
                    if len(slice_index) == 0:
                        continue
                    assert (len(slice_prev_index) != 0) or (
                                len(slice_next_index) != 0), "both previous and next slice are missing"
                    if len(slice_prev_index) == 0:
                        slice_prev_index = slice_next_index
                    if len(slice_next_index) == 0:
                        slice_next_index = slice_prev_index
                    slice_mean_image_data = (flat_image_data[slice_index] + flat_image_data[slice_prev_index] +
                                             flat_image_data[slice_next_index]) / 3
                    slice_ads.extend(np.abs(flat_image_data[slice_index] - slice_mean_image_data))
        """




    def calculate_abs_pairwise_diff(self, flat_image_data):
        num_slices = flat_image_data.shape[0]
        ads = []
        for i in range(num_slices):
            if i % 100 == 0:
                print(f"Calculating for slice {i}")
            comp_slice_indices = list(range(i + 1, num_slices))
            ad = torch.abs(flat_image_data[i] - flat_image_data[comp_slice_indices])
            ads.extend(ad)
        return ads

    # Function to find duplicates
    def find_duplicate_subarrays(self, array):
        count_dict = {}
        for subarr in map(tuple, array):  # Convert each subarray to a tuple
            if subarr in count_dict:
                count_dict[subarr] += 1
            else:
                count_dict[subarr] = 1

        # Count the number of entries with more than one occurrence
        duplicates_count = sum(1 for count in count_dict.values() if count > 1)
        return duplicates_count

