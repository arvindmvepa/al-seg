from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.metrics import pairwise_distances
from numpy.random import RandomState
import abc
import numpy as np


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
    def __init__(self, X, file_names, metric, seed=None):
        self.X = X
        self.flat_X = self.flatten_X()
        self.file_names = file_names
        self.random_state = RandomState(seed=seed)
        self.name = "kcenter"
        self.features = self.flat_X
        self.metric = metric
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
        assert isinstance(already_selected, list)

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

        return new_batch, max(self.min_distances)

    def get_index_w_im_features(self, extra_feature_index):
        return self.num_im_features + extra_feature_index

