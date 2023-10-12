from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.metrics import pairwise_distances
from numpy.random import RandomState
import abc
import numpy as np
from collections import defaultdict
from functools import partial
from active_learning.data_geometry.dist_metrics import euclidean_w_config


class SamplingMethod(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, X, y, seed, **kwargs):
        self.X = X
        self.seed = seed

    def flatten_X(self):
        shape = self.X.shape
        flat_X = self.X
        if len(shape) > 2:
            flat_X = np.reshape(self.X, (shape[0], np.product(shape[1:])))
        return flat_X

    @abc.abstractmethod
    def select_batch_(self):
        return

    def select_batch(self, **kwargs):
        return self.select_batch_(**kwargs)

    def select_batch_unc_(self, **kwargs):
        return self.select_batch_unc_(**kwargs)

    def to_dict(self):
        return None


class kCenterGreedy(SamplingMethod):
    """Returns points that minimizes the maximum distance of any point to a center.
       Implements the k-Center-Greedy method in
       Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
       Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017
       Distance metric defaults to l2 distance.  Features used to calculate distance
       are either raw features or if a model has transform method then uses the output
       of model.transform(X).
       Can be extended to a robust k centers algorithm that ignores a certain number of
       outlier datapoints.  Resulting centers are solution to multiple integer program.
        """
    def __init__(self, X, cfgs, file_names, metric="euclidean", extra_feature_weight=1.0, seed=None):
        self.X = X
        self.flat_X = self.flatten_X()
        self.file_names = file_names
        self.random_state = RandomState(seed=seed)
        self.name = "kcenter"
        self.features = self.flat_X
        self.num_im_features = self.features.shape[1]
        self.cfgs = self.process_cfgs(cfgs)
        self.features = np.concatenate([self.features, self.cfgs], axis=1)
        self.num_extra_features = self.features.shape[1] - self.num_im_features
        self.phase_starting_index = self.num_im_features
        self.phase_ending_index = self.phase_starting_index + 1
        self.group_starting_index = self.phase_ending_index
        self.group_ending_index = self.group_starting_index + self.num_groups
        self.height_starting_index = self.group_ending_index
        self.height_ending_index = self.height_starting_index + 1
        self.weight_starting_index = self.height_ending_index
        self.weight_ending_index = self.weight_starting_index + 1
        self.slice_pos_starting_index = self.weight_ending_index
        self.slice_pos_ending_index = self.slice_pos_starting_index + 1
        self.extra_feature_weight = extra_feature_weight

        if metric == "euclidean_w_config":
            self.metric = partial(euclidean_w_config, num_im_features=self.num_im_features,
                                  phase_starting_index=self.phase_starting_index,
                                  phase_ending_index=self.phase_ending_index,
                                  group_starting_index=self.group_starting_index,
                                  group_ending_index=self.group_ending_index,
                                  height_starting_index=self.height_starting_index,
                                  height_ending_index=self.height_ending_index,
                                  width_starting_index=self.weight_starting_index,
                                  width_ending_index=self.weight_ending_index,
                                  slice_pos_starting_index=self.slice_pos_starting_index,
                                  slice_pos_ending_index=self.slice_pos_ending_index,
                                  extra_feature_weight=self.extra_feature_weight)
        else:
            self.metric = metric
        print(f"Using {metric} as distance metric.")
        self.min_distances = None
        self.max_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.
        Args:
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          rest_dist: whether to reset min_distances.
        """

        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [
                d for d in cluster_centers if d not in self.already_selected
            ]
        if cluster_centers:
            x = self.features[cluster_centers]
            # Update min_distances for all examples given new cluster center.
            dist = pairwise_distances(
                self.features, x, metric=self.metric
            )  # ,n_jobs=4)

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch_(self, already_selected, N, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size
        Returns:
          indices of points selected to minimize distance to cluster centers
        """

        try:
            print("Getting features...")
            print("Calculating distances...")
            self.update_distances(already_selected, only_new=False, reset_dist=True)
        except:
            print("Getting features...")
            print("Calculating distances...")
            self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []

        for i in range(N):
            if (not self.already_selected) and (i == 0):
                # Initialize centers with a randomly selected datapoint
                ind = self.random_state.choice(np.arange(self.n_obs))
            else:
                ind = np.argmax(self.min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected

            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        print(
            "Maximum distance from cluster centers is %0.2f" % max(self.min_distances)
        )

        self.already_selected = already_selected

        return new_batch

    def process_cfgs(self, cfgs):
        # calculate number of slices per frame
        num_slices_dict = dict()
        frame_prefix = "frame"
        frame_num_len = 2
        frame_and_num_prefix_len = len(frame_prefix) + frame_num_len
        for file_name in self.file_names:
            frame_and_num_end_index = file_name.index(frame_prefix) + frame_and_num_prefix_len
            frame_and_num_str = file_name[:frame_and_num_end_index]
            if frame_and_num_str not in num_slices_dict:
                num_slices_dict[frame_and_num_str] = 1
            else:
                num_slices_dict[frame_and_num_str] += 1

        # calculate number of groups
        groups_dict = defaultdict(lambda: len(groups_dict))
        for im_cfg in cfgs:
            groups_dict[im_cfg['Group']]
        self.num_groups = len(groups_dict)
        one_hot_group = [0] * self.num_groups

        # calculate z-score params for height and weight
        height_mean = 0
        height_sstd = 0
        weight_mean = 0
        weight_sstd = 0
        for im_cfg in cfgs:
            height_mean += im_cfg['Height']
            weight_mean += im_cfg['Weight']
        height_mean /= len(cfgs)
        weight_mean /= len(cfgs)

        for im_cfg in cfgs:
            height_sstd += (im_cfg['Height'] - height_mean) ** 2
            weight_sstd += (im_cfg['Weight'] - weight_mean) ** 2
        height_sstd = (height_sstd**(.5))/(len(cfgs) - 1)
        weight_sstd = (weight_sstd**(.5))/(len(cfgs) - 1)



        # encode all cfg features
        slice_str = "slice_"
        slice_str_len = len(slice_str)
        slice_num_len = 1
        extra_features_lst = []
        for im_features, im_cfg, file_name in zip(self.flat_X, cfgs, self.file_names):
            extra_features = []
            # add if frame is ED or ES (one hot encoded)
            frame_and_num_str = file_name[:frame_and_num_end_index]
            frame_num = int(frame_and_num_str[-frame_num_len:])
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

            # add Height and Width
            z_score_height = (im_cfg['Height'] - height_mean) / height_sstd
            z_score_weight = (im_cfg['Weight'] - weight_mean) / weight_sstd
            extra_features.append(z_score_height)
            extra_features.append(z_score_weight)

            # add relative slice position
            slice_start_index = file_name.index(slice_str) + slice_str_len
            slice_end_index = slice_start_index + slice_num_len
            slice_num = int(file_name[slice_start_index:slice_end_index])
            extra_features.append(slice_num / num_slices_dict[frame_and_num_str])

            extra_features_lst.append(np.array(extra_features))

        return np.array(extra_features_lst)

