import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from active_learning.data_geometry.base_coreset import BaseCoreset
from active_learning.data_geometry.gcn import GCN


class CoreGCN(BaseCoreset):
    """Class for identifying representative data points using Coreset sampling"""

    def __init__(self, subset_size="all", hidden_units=128, dropout_rate=0.3, lr_gcn=1e-3, wdecay=5e-4, lambda_loss=1.2,
                 num_epochs_gcn=200, feature_model="resnet18", s_margin=0.1, starting_sample=5, **kwargs):
        super().__init__(feature_model=feature_model, **kwargs)
        assert hasattr(self, "feature_model"), "Feature_model must be defined for CoreGCN"
        self.subset_size = subset_size
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.lr_gcn = lr_gcn
        self.wdecay = wdecay
        self.lambda_loss = lambda_loss
        self.num_epochs_gcn = num_epochs_gcn
        self.s_margin = s_margin
        self.starting_sample = starting_sample
        
    def calculate_representativeness(self, im_score_file, num_samples, round_num, already_selected=[], skip=False,
                                     **kwargs):
        if skip:
            print("Skipping Calculating CoreGCN!")
            return
        sample_indices = []
        already_selected = already_selected.copy()
        if len(already_selected) < self.starting_sample:
            print(f"Calculating KCenterGreedyCoreset until we obtain {self.starting_sample} samples..")
            already_selected_indices = [self.all_train_im_files.index(i) for i in already_selected]
            num_samples_coreset = min(self.starting_sample - len(already_selected), num_samples)
            sample_indices += self.basic_coreset_alg.select_batch_(already_selected=already_selected_indices,
                                                                   N=num_samples_coreset)
            num_samples = num_samples - num_samples_coreset
            # add the just labeled samples to already_selected
            already_selected += [self.all_train_im_files[i] for i in sample_indices]
        if num_samples > 0:
            print("Calculating CoreGCN..")
            all_indices = np.arange(len(self.all_train_im_files))
            already_selected_indices = [self.all_train_im_files.index(i) for i in already_selected]
            unlabeled_indices = np.setdiff1d(all_indices, already_selected_indices)
            if self.subset_size == "all":
                subset_size = len(unlabeled_indices)
            else:
                subset_size = self.subset_size
            assert subset_size <= len(unlabeled_indices), "subset_size must be less than the number of unlabeled indices"
            subset = self.random_state.choice(unlabeled_indices, subset_size, replace=False).tolist()
            binary_labels = torch.cat((torch.zeros([subset_size, 1]),
                                       (torch.ones([len(already_selected_indices), 1]))), 0)
            features = self.image_features[subset+already_selected_indices].to(self.gpus)
            features = nn.functional.normalize(features)
            adj = self.aff_to_adj(features)

            gcn_module = GCN(nfeat=features.shape[1],
                             nhid=self.hidden_units,
                             nclass=1,
                             dropout=self.dropout_rate).to(self.gpus)

            models = {'gcn_module': gcn_module}

            optim_backbone = optim.Adam(models['gcn_module'].parameters(), lr=self.lr_gcn, weight_decay=self.wdecay)
            optimizers = {'gcn_module': optim_backbone}

            nlbl = np.arange(0, subset_size, 1)
            lbl = np.arange(subset_size, subset_size + len(already_selected_indices), 1)

            ############
            print("Training GCN..")
            for i in range(self.num_epochs_gcn):
                optimizers['gcn_module'].zero_grad()
                outputs, _, _ = models['gcn_module'](features, adj)
                lamda = self.lambda_loss
                loss = self.BCEAdjLoss(outputs, lbl, nlbl, lamda)
                loss.backward()
                optimizers['gcn_module'].step()
                if i % 50 == 0:
                    print("GCN, Epoch: ", i, "Loss: ", loss.item())

            models['gcn_module'].eval()
            print("Getting GCN features...")
            with torch.no_grad():
                inputs = features.to(self.gpus)
                labels = binary_labels.to(self.gpus)
                scores, _, feat = models['gcn_module'](inputs, adj)

                feat = feat.detach().cpu().numpy()
                coreset_inst = self.create_coreset_inst(feat)
                sample_indices += coreset_inst.select_batch_(lbl, num_samples)

                print("Max confidence value: ", torch.max(scores.data).item())
                print("Mean confidence value: ", torch.mean(scores.data).item())
                preds = torch.round(scores)
                correct_labeled = (preds[subset_size:, 0] == labels[subset_size:, 0]).sum().item() / len(already_selected_indices)
                correct_unlabeled = (preds[:subset_size, 0] == labels[:subset_size, 0]).sum().item() / subset_size
                correct = (preds[:, 0] == labels[:, 0]).sum().item() / (subset_size + len(already_selected_indices))
                print("Labeled classified %: ", correct_labeled)
                print("Unlabeled classified %: ", correct_unlabeled)
                print("Total correctly classified %: ", correct)

        # write score file
        with open(im_score_file, "w") as f:
            # higher score for earlier added images
            scores = [score for score in range(len(sample_indices), 0, -1)]
            for i, im_file in enumerate(self.all_train_im_files):
                if i in sample_indices:
                    score = scores[sample_indices.index(i)]
                else:
                    score = 0
                f.write(f"{im_file},{score}\n")

        return [self.all_train_im_files[i] for i in sample_indices]

    def aff_to_adj(self, x, y=None, eps=1e-10):
        x = x.detach().cpu().numpy()
        adj = np.matmul(x, x.transpose())
        adj += -1.0 * np.eye(adj.shape[0])
        adj_diag = np.sum(adj, axis=0)  # rowise sum
        adj = np.matmul(adj, np.diag(1 / (adj_diag + eps)))
        adj = adj + np.eye(adj.shape[0])
        adj = torch.Tensor(adj).to(self.gpus)

        return adj

    def BCEAdjLoss(self, scores, lbl, nlbl, l_adj, eps=1e-10):
        lnl = torch.log(scores[lbl] + eps)
        lnu = torch.log(1 - scores[nlbl] + eps)
        labeled_score = torch.mean(lnl)
        unlabeled_score = torch.mean(lnu)
        bce_adj_loss = -labeled_score - l_adj*unlabeled_score
        return bce_adj_loss
