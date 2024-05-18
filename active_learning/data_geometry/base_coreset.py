from glob import glob
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
                 uncertainty_score_file="entropy.txt", use_labels=False, ann_type=None, label_wt=1.0,
                 default_label_val=1, max_dist=None, pos_wt=1.0, normalize_pos_by_label_ct=False, wt_max_dist_mult=1.0,
                 extra_feature_wt=0.0, patient_wt=0.0, phase_wt=0.0, group_wt=0.0, height_wt=0.0, weight_wt=0.0,
                 slice_rel_pos_wt=0.0, slice_mid_wt=0.0, slice_pos_wt=0.0, uncertainty_wt=0.0, feature_model=False,
                 feature_model_params=None, contrastive=False, use_model_features=False, delete_preds=True,
                 use_model_fts_as_pos_fts=False, seed=0, gpus="cuda:0", **kwargs):
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
        self.default_label_val = default_label_val
        self.max_dist = max_dist
        self.pos_wt = pos_wt
        self.normalize_pos_by_label_ct = normalize_pos_by_label_ct
        self.wt_max_dist_mult = wt_max_dist_mult
        self.contrastive = contrastive
        self.gpus = gpus
        self.feature_model = feature_model
        self.fuse_image_data = self.feature_model_params.get("fuse_image_data", False)
        self.use_model_features = use_model_features
        self.delete_preds = delete_preds
        self.use_model_fts_as_pos_fts = use_model_fts_as_pos_fts
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
        self.dataset = DatasetFactory.create_dataset(dataset_type=self.dataset_type,
                                                     all_train_im_files=self.all_train_im_files,
                                                     all_train_full_im_paths=self.all_train_full_im_paths,
                                                     **self.dataset_kwargs)
        self.setup_image_features()
        print("Done setting up data")

    def setup_image_features(self):
        print("Setting up image features...")
        print("Getting data")
        image_data, self.image_meta_data, self.image_labels_arr  =  self.dataset.get_data()
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

    def create_coreset_inst(self, processed_data, prev_round_dir=None, train_logits_path=None, uncertainty_kwargs=None):
        print("Creating coreset instance...")
        print("Getting coreset metric and features...")
        coreset_metric, features = self.get_coreset_metric_and_features(processed_data, prev_round_dir=prev_round_dir,
                                                                        train_logits_path=train_logits_path,
                                                                        uncertainty_kwargs=uncertainty_kwargs)
        coreset_inst = self.coreset_cls(X=features, file_names=self.all_train_im_files, metric=coreset_metric,
                                        num_im_features=processed_data.shape[1],
                                        uncertainty_starting_index=self.non_image_indices["uncertainty_starting_index"] if self.use_uncertainty else None,
                                        uncertainty_ending_index=self.non_image_indices["uncertainty_ending_index"] if self.use_uncertainty else None,
                                        uncertainty_wt=self.non_image_wts["uncertainty_wt"] if self.use_uncertainty else None,
                                        gpus=self.gpus, **self.coreset_kwargs)
        print(f"Created {coreset_inst.name} inst!")
        return coreset_inst, features


    def get_coreset_inst_and_features_for_round(self, prev_round_dir, train_logits_path=None, uncertainty_kwargs=None):
        feat = self.get_features()
        coreset_inst, feat = self.create_coreset_inst(feat, prev_round_dir=prev_round_dir,
                                                      train_logits_path=train_logits_path,
                                                      uncertainty_kwargs=uncertainty_kwargs)
        return coreset_inst, feat

    def get_model_features(self, prev_round_dir, train_logits_path):
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
        if self.delete_preds:
            for train_result_ in train_results:
                os.remove(train_result_)

        return preds_arrs

    def get_coreset_metric_and_features(self, processed_data, prev_round_dir=None, train_logits_path=None, uncertainty_kwargs=None):
        if self.use_model_features or self.use_model_fts_as_pos_fts:
            print("Using Model Features")
            model_features_exist = True
            model_features = self.get_model_features(prev_round_dir, train_logits_path)
            if model_features is None:
                print("Model features not found. Using image features instead")
                im_features = processed_data
                model_features_exist = False
            elif self.use_model_fts_as_pos_fts:
                print("Combining model (pos features) and image features")
                print(f"image features shape: {processed_data.shape}")
                print(f"Model features shape: {model_features.shape}")
                im_features = np.concatenate([model_features, processed_data], axis=1)
                print(f"Combined features shape: {im_features.shape}")
            else:
                print(f"Use model features alone. Model features shape: {model_features.shape}")
                im_features = model_features
        else:
            im_features = processed_data
        num_im_features = im_features.shape[1]

        # check that number of pixels in image is greater than number of labels
        # only update points that are part of the image features
        if (self.use_model_fts_as_pos_fts and model_features_exist and self.use_labels) or (not self.use_model_fts_as_pos_fts and self.use_labels):
            print(f"Using labels with weight {self.label_wt}")
            if self.use_model_fts_as_pos_fts:
                print("Old labels shape: ", self.image_labels_arr.shape)
                labels = np.repeat(self.image_labels_arr[..., None], repeats=4, axis=-1)
                print("Reshape labels shape: ", labels.shape)
                labels = labels.reshape(im_features.shape[0], -1)
                print("New labels shape: ", self.image_labels_arr.shape)
                num_label_features = labels.shape[1]
            else:
                labels = self.image_labels_arr.reshape(im_features.shape[0], -1)
                num_label_features = labels.shape[1]
            assert num_im_features >= num_label_features
            # hard-coded number of classes to be (background + 3)
            print(f"Using default label val {self.default_label_val}")
            label_mask = np.where(labels < 4, self.label_wt, self.default_label_val)
            features = np.concatenate([im_features, label_mask, self.image_meta_data_arr], axis=1)
        # hard code 1 labels for hacky fix to the metric to keep track of position
        elif self.fuse_image_data:
            labels = self.image_labels_arr.reshape(im_features.shape[0], -1)
            num_label_features = labels.shape[1]
            assert num_im_features >= num_label_features
            label_mask = np.ones(labels.shape)
            features = np.concatenate([im_features, label_mask, self.image_meta_data_arr], axis=1)
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
        print("Using pos_wt: ", self.pos_wt)
        print("Using normalize_pos_by_label_ct: ", self.normalize_pos_by_label_ct)
        coreset_metric = partial(metric_w_config, image_metric=self.metric, max_dist=self.max_dist,
                                 num_im_features=num_im_features, num_label_features=num_label_features,
                                 pos_wt=self.pos_wt, normalize_pos_by_label_ct=self.normalize_pos_by_label_ct,
                                 **non_image_kwargs)
        return coreset_metric, features

    def _update_non_image_wts(self, extra_feature_wt=None, patient_wt=None, phase_wt=None, group_wt=None, height_wt=None,
                              weight_wt=None, slice_rel_pos_wt=None, slice_mid_wt=None, slice_pos_wt=None, uncertainty_wt=None,
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
