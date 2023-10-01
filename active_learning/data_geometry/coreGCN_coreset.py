import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch
import torch.optim as optim
from active_learning.data_geometry.base_coreset import BaseCoreset, CoresetDatasetWrapper
from active_learning.data_geometry.gcn import GCN


class CoreGCNCoreset(BaseCoreset):
    """Class for identifying representative data points using Coreset sampling"""

    def __init__(self, patch_size=(256, 256), batch_size=128, hidden_units=128, dropout_rate=0.3, lr_gcn=1e-3,
                 wdecay=5e-4, lambda_loss=1.2, gpus="cuda:0",
                 **kwargs):
        super().__init__(alg_string="kcenter_greedy", patch_size=patch_size)
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.lr_gcn = lr_gcn
        self.wdecay = wdecay
        self.lambda_loss = lambda_loss
        self.gpus = gpus
        
    def calculate_representativeness(self, im_score_file, num_samples, already_selected=[], skip=False, **kwargs):
        if skip:
            print("Skipping Calculating CoreGCN!")
            return

        print("Calculating CoreGCN..")
        all_indices = np.arange(len(self.all_train_im_files))
        already_selected_indices = [self.all_train_im_files.index(i) for i in already_selected]
        unlabeled_indices = np.setdiff1d(all_indices, already_selected_indices)
        subset = self.random_state.choice(unlabeled_indices, num_samples, replace=False)
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                 sampler=SubsetSequentialSampler(subset+already_selected_indices), pin_memory=True)
        binary_labels = torch.cat((torch.zeros([num_samples, 1]),
                                   (torch.ones([len(already_selected_indices), 1]))), 0)
        features = self.get_features(data_loader)
        features = nn.functional.normalize(features)
        adj = self.aff_to_adj(features)

        gcn_module = GCN(nfeat=features.shape[1],
                         nhid=self.hidden_units,
                         nclass=1,
                         dropout=self.dropout_rate).to(self.gpus)

        models = {'gcn_module': gcn_module}

        optim_backbone = optim.Adam(models['gcn_module'].parameters(), lr=self.lr_gcn, weight_decay=self.wdecay)
        optimizers = {'gcn_module': optim_backbone}

        nlbl = np.arange(0, num_samples, 1)
        lbl = np.arange(num_samples, num_samples + len(already_selected_indices), 1)

        ############
        for _ in range(200):
            optimizers['gcn_module'].zero_grad()
            outputs, _, _ = models['gcn_module'](features, adj)
            lamda = self.lambda_loss
            loss = self.BCEAdjLoss(outputs, lbl, nlbl, lamda)
            loss.backward()
            optimizers['gcn_module'].step()

        models['gcn_module'].eval()
        with torch.no_grad():
            inputs = features.to(self.gpus)
            labels = binary_labels.to(self.gpus)
            scores, _, feat = models['gcn_module'](inputs, adj)

            # TODO: stopped here - continue here.
            if method == "CoreGCN":
                feat = feat.detach().cpu().numpy()
                new_av_idx = np.arange(SUBSET, (SUBSET + (cycle + 1) * ADDENDUM))
                sampling2 = kCenterGreedy(feat)
                batch2 = sampling2.select_batch_(new_av_idx, ADDENDUM)
                other_idx = [x for x in range(SUBSET) if x not in batch2]
                arg = other_idx + batch2

            else:

                s_margin = args.s_margin
                scores_median = np.squeeze(torch.abs(scores[:SUBSET] - s_margin).detach().cpu().numpy())
                arg = np.argsort(-(scores_median))

            print("Max confidence value: ", torch.max(scores.data))
            print("Mean confidence value: ", torch.mean(scores.data))
            preds = torch.round(scores)
            correct_labeled = (preds[SUBSET:, 0] == labels[SUBSET:, 0]).sum().item() / ((cycle + 1) * ADDENDUM)
            correct_unlabeled = (preds[:SUBSET, 0] == labels[:SUBSET, 0]).sum().item() / SUBSET
            correct = (preds[:, 0] == labels[:, 0]).sum().item() / (SUBSET + (cycle + 1) * ADDENDUM)
            print("Labeled classified: ", correct_labeled)
            print("Unlabeled classified: ", correct_unlabeled)
            print("Total classified: ", correct)


        # write score file
        with open(im_score_file, "w") as f:
            # higher score for earlier added images
            scores = [score for score in range(len(core_set_indices), 0, -1)]
            for i, im_file in enumerate(self.all_train_im_files):
                if i in core_set_indices:
                    score = scores[core_set_indices.index(i)]
                else:
                    score = 0
                f.write(f"{im_file},{score}\n")

        return [self.all_train_im_files[i] for i in core_set_indices]

    def setup(self, data_root, all_train_im_files):
        super().setup(data_root, all_train_im_files)
        self.dataset = CoresetDatasetWrapper(self.all_processed_train_data, transform=T.ToTensor())

    def get_features(self, data_loader):
        """For now just a pass through, not using models"""
        features = torch.tensor([]).to(self.gpus)
        with torch.no_grad():
            for inputs, _, _ in data_loader:
                inputs = inputs.to(self.gpus)
                features = torch.cat((features, inputs), 0)
            feat = features
        return feat

    def aff_to_adj(self, x, y=None):
        x = x.detach().cpu().numpy()
        adj = np.matmul(x, x.transpose())
        adj += -1.0 * np.eye(adj.shape[0])
        adj_diag = np.sum(adj, axis=0)  # rowise sum
        adj = np.matmul(adj, np.diag(1 / adj_diag))
        adj = adj + np.eye(adj.shape[0])
        adj = torch.Tensor(adj).cuda()

        return adj

    def BCEAdjLoss(self, scores, lbl, nlbl, l_adj):
        lnl = torch.log(scores[lbl])
        lnu = torch.log(1 - scores[nlbl])
        labeled_score = torch.mean(lnl)
        unlabeled_score = torch.mean(lnu)
        bce_adj_loss = -labeled_score - l_adj*unlabeled_score
        return bce_adj_loss


class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)