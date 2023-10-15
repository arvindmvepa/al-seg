import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from active_learning.data_geometry.base_coreset import BaseCoreset
from active_learning.data_geometry.gcn import GCN
from tqdm import tqdm


class CoreGCN(BaseCoreset):
    """Class for identifying representative data points using Coreset sampling"""

    def __init__(self, subset_size="all", hidden_units=128, dropout_rate=0.3, lr_gcn=1e-3, wdecay=5e-4, lambda_loss=1.2,
                 num_epochs_gcn=200, s_margin=0.1, starting_sample=5, adj_sim_wt_metric=None, **kwargs):
        super().__init__(**kwargs)
        self.subset_size = subset_size
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.lr_gcn = lr_gcn
        self.wdecay = wdecay
        self.lambda_loss = lambda_loss
        self.num_epochs_gcn = num_epochs_gcn
        self.s_margin = s_margin
        self.starting_sample = starting_sample
        if adj_sim_wt_metric is None:
            self.adj_sim_wt_metric = None
        elif adj_sim_wt_metric not in wt_metrics:
            raise ValueError(f"adj_sim_wt_metric must be None or one of {list(wt_metrics.keys())}. Instead got {adj_sim_wt_metric}")
        else:
            self.adj_sim_wt_metric = wt_metrics[adj_sim_wt_metric]
        
    def calculate_representativeness(self, im_score_file, num_samples, prev_round_dir, train_logits_path,
                                     already_selected=[], skip=False, delete_preds=True, **kwargs):
        if skip:
            print("Skipping Calculating CoreGCN!")
            return
        coreset_inst, feat = self.get_coreset_inst_and_features_for_round(prev_round_dir, train_logits_path,
                                                                          delete_preds=delete_preds)
        sample_indices = []
        already_selected = already_selected.copy()
        if len(already_selected) < self.starting_sample:
            print(f"Calculating KCenterGreedyCoreset until we obtain {self.starting_sample} samples..")
            already_selected_indices = [self.all_train_im_files.index(i) for i in already_selected]
            num_samples_coreset = min(self.starting_sample - len(already_selected), num_samples)
            sample_indices += coreset_inst.select_batch_(already_selected=already_selected_indices,
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
            if isinstance(feat, np.ndarray):
                features = torch.from_numpy(feat[subset + already_selected_indices]).float().to(self.gpus)
            else:
                raise ValueError("feat must be a numpy array")
            features = nn.functional.normalize(features)
            print("Getting adjacency matrix...")
            adj = self.aff_to_adj(features)
            print("Finished.")

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
        num_ims = len(self.all_train_im_files)
        print(f"num_ims: {num_ims}, x.shape[0]: {x.shape[0]}")
        assert num_ims > 1, "Number of images must be greater than 1"
        num_features = x.shape[1]
        assert num_features > 1, "Number of features must be greater than 1"
        if self.adj_sim_wt_metric is not None:
            adj = np.eye(num_features)
            for i in tqdm(range(num_ims)):
                slice_no = self.get_slice_no(i)
                cur_index = i + 1
                cur_slice_no = self.get_slice_no(i)
                while cur_slice_no != 0 and cur_index < num_ims:
                    wt = self.adj_sim_wt_metric(slice_no, cur_slice_no)
                    adj[i, cur_index] = wt
                    adj[cur_index, i] = wt
                    cur_index += 1
                    cur_slice_no = self.get_slice_no(i)
        else:
            x = x.detach().cpu().numpy()
            adj = np.matmul(x, x.transpose())
        adj += -1.0 * np.eye(adj.shape[0])
        adj_diag = np.sum(adj, axis=0)  # rowise sum
        adj = np.matmul(adj, np.diag(1 / (adj_diag + eps)))
        adj = adj + np.eye(adj.shape[0])
        adj = torch.Tensor(adj).to(self.gpus)

        return adj

    def get_slice_no(self, image_index):
        if (0 <= image_index) and (image_index >= self.image_cfgs_arr.shape[0]):
            return None
        else:
            return self.image_cfgs_arr[image_index, self.slice_pos_starting_index:self.slice_pos_ending_index]

    def BCEAdjLoss(self, scores, lbl, nlbl, l_adj, eps=1e-10):
        lnl = torch.log(scores[lbl] + eps)
        lnu = torch.log(1 - scores[nlbl] + eps)
        labeled_score = torch.mean(lnl)
        unlabeled_score = torch.mean(lnu)
        bce_adj_loss = -labeled_score - l_adj*unlabeled_score
        return bce_adj_loss


def get_k_slice_wt(orig_slice_pos, other_slice_pos, k=1):
    if np.abs(orig_slice_pos - other_slice_pos) <= k:
        return 1
    else:
        return 0

def get_inv_slice_wt(orig_slice_pos, other_slice_pos):
    if orig_slice_pos == other_slice_pos:
        return 1
    else:
        return 1/(np.abs(orig_slice_pos - other_slice_pos))

def get_exp_slice_wt(orig_slice_pos, other_slice_pos):
    if orig_slice_pos == other_slice_pos:
        return 1
    else:
        return 1/(np.exp(np.abs(orig_slice_pos - other_slice_pos) -1))


wt_metrics = {"k_slice_wt": get_k_slice_wt,
              "inv_slice_wt": get_inv_slice_wt,
              "exp_slice_wt": get_exp_slice_wt}
