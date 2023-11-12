from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.metrics import pairwise_distances
from numpy.random import RandomState
import abc
import numpy as np
import torch


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
    def __init__(self, X, file_names, metric, num_im_features, uncertainty_index=None, uncertainty_wt=None, seed=None,
                 gpus="cuda:0"):
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
        max_dist = max(self.min_distances)
        print(
            "Maximum distance from cluster centers is %0.2f" % max_dist
        )

        self.already_selected = already_selected

        return new_batch, max_dist


class GPUkCenterGreedy(SamplingMethod):
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
    def __init__(self, X, file_names, metric, num_im_features, uncertainty_starting_index=None,
                 uncertainty_ending_index=None, uncertainty_wt=None, seed=None,
                 gpus="cuda:0"):
        self.X = X
        self.gpus = gpus
        self.num_im_features = num_im_features
        if (isinstance(uncertainty_starting_index, int) and uncertainty_starting_index > 0) and (isinstance(uncertainty_ending_index, int) and uncertainty_ending_index > 0) :
            self.uncertainty_starting_index = uncertainty_starting_index
            self.uncertainty_ending_index = uncertainty_ending_index
            self.uncertainty_wt = uncertainty_wt
            self.use_uncertainty = True
            print("Using uncertainty in GPUKCenterGreedy")
        else:
            self.uncertainty_starting_index = None
            self.uncertainty_ending_index = None
            self.uncertainty_wt = None
            self.use_uncertainty = False
            print("Not using uncertainty in GPUKCenterGreedy")
        self.flat_X, self.wt_uncertainty_arr = self.extract_and_flatten_X()
        self.file_names = file_names
        self.random_state = RandomState(seed=seed)
        self.name = "kcenter"
        self.features = self.flat_X
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
            print("Starting to calculate pairwise distances...")
            dist = torch.cdist(torch.unsqueeze(self.features, 0), torch.unsqueeze(x, 0), p=2).squeeze(0)
            print("Done calculating pairwise distances. dist.shape = ", dist.shape)
            print("Starting to add in uncertainty...")

            if self.use_uncertainty:
                # dimensions will automatically broadcoast
                dist = dist + self.wt_uncertainty_arr * 0.5
                for j in range(x.shape[0]):
                    dist[:,j] = dist[:, j] + self.wt_uncertainty_arr[j]*0.5
            print("Done adding in uncertainty")
            min_dist = torch.min(dist, axis=1)
            if self.min_distances is None:
                self.min_distances = torch.reshape(min_dist, (-1, 1))
            else:
                self.min_distances = torch.minimum(self.min_distances, min_dist)

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
        max_dist = max(self.min_distances)
        print(
            "Maximum distance from cluster centers is %0.2f" % max_dist
        )

        self.already_selected = already_selected

        return new_batch, max_dist

    def extract_and_flatten_X(self):
        im_features = self.X[:, :self.num_im_features]
        uncertainty_arr = None
        if self.use_uncertainty:
            uncertainty_arr = np.sum(self.X[:, self.uncertainty_starting_index:self.uncertainty_ending_index], axis=1) * self.uncertainty_wt
            print("Using uncertainty, uncertainty array shape: ", uncertainty_arr.shape)
            uncertainty_arr = torch.from_numpy(uncertainty_arr).float().to(self.gpus)
        shape = im_features.shape
        flat_X = im_features
        if len(shape) > 2:
            flat_X = np.reshape(im_features, (shape[0], np.product(shape[1:])))
        flat_X = torch.from_numpy(flat_X).float().to(self.gpus)
        return flat_X, uncertainty_arr


class ProbkCenterGreedy(kCenterGreedy):
    def __init__(self, temp_init=1.0, temp_scale=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "prob_kcenter"
        self.temp_init = temp_init
        self.temp_scale = temp_scale
        print("Initialized ProbkCenterGreedy!")

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
        # set iter based on how many points have already been selected
        start_iter = len(combined_already_selected)
        end_iter = start_iter + N
        temp = None
        print(f"Starting iteration {start_iter}, temperature {self.get_temp(start_iter)}")
        for iter in range(start_iter, end_iter):
            all_indices = np.arange(self.n_obs)
            if not combined_already_selected and (iter == 0):
                # Initialize centers with a randomly selected datapoint
                ind = self.random_state.choice(all_indices)
            else:
                unselected_mask = np.ones(self.min_distances.shape[0], dtype=bool)
                unselected_mask[already_selected] = False
                min_distances = self.min_distances[unselected_mask]
                temp = self.get_temp(iter)
                unselected_probs = np.squeeze(self.generate_probs(min_distances, temp))
                unselected_indices = all_indices[unselected_mask]
                ind = self.random_state.choice(unselected_indices, p=unselected_probs)
            assert ind not in already_selected
            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        print(f"Ending iteration {iter}, temperature {temp}")
        max_dist = max(self.min_distances)
        print(
            "Maximum distance from cluster centers is %0.2f" % max_dist
        )

        self.already_selected = already_selected

        return new_batch, max_dist

    def get_temp(self, iter):
        if self.temp_scale == "inv_iter":
            return self.temp_init / (iter + 1)
        else:
            return self.temp_init


    def generate_probs(self, min_distances, temp):
        probs = self.softmax(min_distances, temp)
        return probs

    @staticmethod
    def softmax(x, temp):
        if temp <= 0:
            raise ValueError("Temperature must be a positive scalar.")

        # Shift the logits to avoid numerical overflows.
        x_shifted = x - np.max(x)

        # Compute the exponentials of the shifted logits.
        exps = np.exp(x_shifted / temp)

        return exps / np.sum(exps)

