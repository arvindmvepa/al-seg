# https://github.com/avihu111/TypiClust/blob/0e24b5f625adb2f3fb96920c2952707e82691dd0/deep-al/pycls/al/typiclust.py

import torch 
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.neighbors import NearestNeighbors
import h5py
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
import numpy as np
from tqdm import tqdm 
from active_learning.data_geometry.base_typiclust import BaseTypiclust
from active_learning.data_geometry.vaal import VAE, Discriminator, AdversarySampler


class Typiclust(BaseTypiclust):
    """Class for identifying representative data points using Coreset sampling. Based on https://github.com/sinhasam/vaal/blob/master/main.py#L57"""

    def __init__(self, min_cluster_size=5, max_num_clusters=500, k_nn=20, knn_model='sklearn',
        **kwargs):
        super().__init__(**kwargs)
        """knn_model can be 'sklearn' or 'faiss'"""
        self.min_cluster_size = min_cluster_size
        self.max_num_clusters = max_num_clusters
        self.k_nn = k_nn
        self.knn_model = knn_model

    def calculate_representativeness(self, im_score_file, num_samples, already_selected=[], skip=False, **kwargs):
        if skip:
            print("Skipping Calculating Typiclust!")
            return
        print("Calculating Representative Clustering..")
        all_indices = np.arange(len(self.all_train_im_files))
        already_selected_indices = [self.all_train_im_files.index(i) for i in already_selected]
        unlabeled_indices = np.setdiff1d(all_indices, already_selected_indices)
        
        num_clusters = min(len(already_selected_indices) + num_samples, self.max_num_clusters)
        print(f'Clustering into {num_clusters} clusters..')
        clusters = kmeans(self.features, num_clusters=num_clusters, random_state=self.seed)
        print(f'Finished clustering into {num_clusters} clusters.')
        
        print("Start selecting representative samples..")
        activeSet, _ = self.select_samples(knn_model=self.knn_model,
                                           lSet=already_selected_indices, 
                                           uSet=unlabeled_indices, 
                                           clusters=clusters, 
                                           budgetSize=num_samples)
        activeSet = activeSet.tolist()
        
        # write score file
        with open(im_score_file, "w") as f:
            # higher score for earlier added images
            scores = [score for score in range(len(activeSet), 0, -1)]
            for i, im_file in enumerate(self.all_train_im_files):
                if i in activeSet:
                    score = scores[activeSet.index(i)]
                else:
                    score = 0
                f.write(f"{im_file},{score}\n")

        return [self.all_train_im_files[i] for i in activeSet]


    def select_samples(self, knn_model, lSet, uSet, clusters, budgetSize):
        # using only labeled+unlabeled indices, without validation set.
        relevant_indices = np.concatenate([lSet, uSet]).astype(int)
        features = self.features[relevant_indices]
        labels = np.copy(clusters[relevant_indices])
        existing_indices = np.arange(len(lSet))
        # counting cluster sizes and number of labeled samples per cluster
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))
        clusters_df = pd.DataFrame({'cluster_id': cluster_ids, 
                                    'cluster_size': cluster_sizes, 
                                    'existing_count': cluster_labeled_counts,
                                    'neg_cluster_size': -1 * cluster_sizes})
        # drop too small clusters
        clusters_df = clusters_df[clusters_df.cluster_size > self.min_cluster_size]
        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        labels[existing_indices] = -1

        selected = []

        print(f'Getting nearest neighbors using {self.knn_model} knn...')
        bad_clusters = 0
        for i in tqdm(range(budgetSize)):
            cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id
            indices = (labels == cluster).nonzero()[0]
            rel_feats = features[indices]
            if rel_feats.shape[0] == 0:
                bad_clusters += 1
                continue
            # in case we have too small cluster, calculate density among half of the cluster
            typicality = calculate_typicality(knn_model, rel_feats, max(min(self.k_nn, len(indices) // 2),1))
            idx = indices[typicality.argmax()]
            selected.append(idx)
            labels[idx] = -1
        print("len(np.intersect1d(selected, existing_indices)): ", len(np.intersect1d(selected, existing_indices)))
        if bad_clusters > 0:
            remaining_unlabeled = [sample for sample in uSet if sample not in selected]
            addtl_samples = self.random_state.choice(remaining_unlabeled, bad_clusters,
                                                     replace=False)
            print("len(np.intersect1d(addtl_samples, existing_indices)): ", len(np.intersect1d(addtl_samples, existing_indices)))
            selected.extend(addtl_samples.tolist())
            print(f'Had {bad_clusters} clusters with no samples, adding random samples..')
        selected = np.array(selected)
        assert len(selected) == budgetSize, 'added a different number of samples'
        assert len(np.intersect1d(selected, existing_indices)) == 0, 'should be new samples'
        activeSet = relevant_indices[selected]
        remainSet = np.array(sorted(list(set(uSet) - set(activeSet))))
        
        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        return activeSet, remainSet
    

# def get_nn(features, num_neighbors):
#     # calculates nearest neighbors on GPU
#     d = features.shape[1]
#     features = features.astype(np.float32)
#     cpu_index = faiss.IndexFlatL2(d)
#     # Below needs a CUDA enabled GPU
#     try:
#         gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
#         gpu_index.add(features)  # add vectors to the index
#         distances, indices = gpu_index.search(features, num_neighbors + 1)
#     except: # Use CPU if GPU is not available
#         cpu_index.add(features)  # Add vectors to the index (on CPU)
#         distances, indices = cpu_index.search(features, num_neighbors + 1)
#     # 0 index is the same sample, dropping it
#     return distances[:, 1:], indices[:, 1:]


def get_nn_sklearn(features, num_neighbors):
    # calculates nearest neighbors with sklearn
    if features.shape[0] == 1:
        num_neighbors = 1
    else:
        num_neighbors = num_neighbors + 1
    nn = NearestNeighbors(n_neighbors=num_neighbors, n_jobs=-1)
    nn.fit(features)
    distances, indices = nn.kneighbors(features)
    return distances[:, 1:], indices[:, 1:]


def get_mean_nn_dist(model, features, num_neighbors, return_indices=False):
    # if model == 'sklearn':
    print("features.shape", features.shape)
    print("num_neighbors", num_neighbors)
    distances, indices = get_nn_sklearn(features, num_neighbors)
    # elif model == 'faiss':
    #     distances, indices = get_nn(features, num_neighbors)
    print("distances.shape", distances.shape)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance


def calculate_typicality(model, features, num_neighbors):
    mean_distance = get_mean_nn_dist(model, features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)
    return typicality


def kmeans(features, num_clusters, random_state=0):
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters, random_state=random_state)
        km.fit_predict(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, random_state=random_state)
        km.fit_predict(features)
    return km.labels_