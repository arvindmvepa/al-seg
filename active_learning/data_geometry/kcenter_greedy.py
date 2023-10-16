from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.metrics import pairwise_distances
from numpy.random import RandomState
import abc
import numpy as np
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
    def __init__(self, X, file_names, cfgs_arr, phase_starting_index, phase_ending_index, group_starting_index,
                 group_ending_index, height_starting_index, height_ending_index, weight_starting_index,
                 weight_ending_index, slice_rel_pos_starting_index, slice_rel_pos_ending_index,
                 slice_pos_starting_index, slice_pos_ending_index,metric="euclidean", extra_feature_weight=1.0, phase_weight=1.0,
                 group_weight=1.0, height_weight=1.0, weight_weight=1.0, slice_pos_weight=1.0, seed=None):
        self.X = X
        self.flat_X = self.flatten_X()
        self.file_names = file_names
        self.random_state = RandomState(seed=seed)
        self.name = "kcenter"
        self.features = self.flat_X

        if metric == "euclidean_w_config":
            self.num_im_features = self.features.shape[1]
            self.features = np.concatenate([self.features, cfgs_arr], axis=1)
            self.num_extra_features = self.features.shape[1] - self.num_im_features
            self.phase_starting_index = self.update_index_w_im_features(phase_starting_index)
            self.phase_ending_index = self.update_index_w_im_features(phase_ending_index)
            self.group_starting_index = self.update_index_w_im_features(group_starting_index)
            self.group_ending_index = self.update_index_w_im_features(group_ending_index)
            self.height_starting_index = self.update_index_w_im_features(height_starting_index)
            self.height_ending_index = self.update_index_w_im_features(height_ending_index)
            self.weight_starting_index = self.update_index_w_im_features(weight_starting_index)
            self.weight_ending_index = self.update_index_w_im_features(weight_ending_index)
            self.slice_rel_pos_starting_index = self.update_index_w_im_features(slice_rel_pos_starting_index)
            self.slice_rel_pos_ending_index = self.update_index_w_im_features(slice_rel_pos_ending_index)
            self.slice_pos_starting_index = self.update_index_w_im_features(slice_pos_starting_index)
            self.slice_pos_ending_index = self.update_index_w_im_features(slice_pos_ending_index)
            self.extra_feature_weight = extra_feature_weight
            self.phase_weight = phase_weight
            self.group_weight = group_weight
            self.height_weight = height_weight
            self.weight_weight = weight_weight
            self.slice_pos_weight = slice_pos_weight
            self.metric = partial(euclidean_w_config, num_im_features=self.num_im_features,
                                  phase_starting_index=self.phase_starting_index,
                                  phase_ending_index=self.phase_ending_index,
                                  group_starting_index=self.group_starting_index,
                                  group_ending_index=self.group_ending_index,
                                  height_starting_index=self.height_starting_index,
                                  height_ending_index=self.height_ending_index,
                                  weight_starting_index=self.weight_starting_index,
                                  weight_ending_index=self.weight_ending_index,
                                  slice_pos_starting_index=self.slice_pos_starting_index,
                                  slice_pos_ending_index=self.slice_pos_ending_index,
                                  extra_feature_weight=self.extra_feature_weight,
                                  phase_weight=self.phase_weight, group_weight=self.group_weight,
                                  height_weight=self.height_weight, weight_weight=self.weight_weight,
                                  slice_pos_weight=self.slice_pos_weight)
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
        except Exception as error:
            print(f"Previous attempt generated error: {error}")
            print("Getting features...")
            print("Calculating distances...")
            self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []

        combined_already_selected = already_selected + self.already_selected
        for i in range(N):
            if not combined_already_selected and (i == 0):
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

    def update_index_w_im_features(self, extra_feature_index):
        return self.num_im_features + extra_feature_index
