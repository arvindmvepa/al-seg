import torch
import torch.nn as nn
from itertools import combinations


class ContrastiveLoss(nn.Module):
    """Contrastive loss: https://ieeexplore.ieee.org/abstract/document/1640964"""
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin -
                                                                      euclidean_distance, min=0.0), 2))
        return loss_contrastive


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature=0.5 ):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # this is matching all the elements with their augmentations (top and bottom of matrix)
        # collecting all the similarities of the positive pairs
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # includes the positive pairs (but don't include main diagonal)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # removes elements on main diagonal as well as off diagonal augmentation pairs
        negative_samples = sim[self.mask].reshape(N, -1)

        # basically a multi-class problem but for each sample
        # the correct label is the first index for all the samples
        labels = torch.zeros(N).to(positive_samples.device).long()

        # this is why positive samples goes first in the concat
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class NT_Xent_Group_Neg(NT_Xent):

    def __init__(self, use_patient=False, use_phase=False, use_slice_pos=False, debug=False, **kwargs):
        assert use_patient or use_phase or use_slice_pos, "At least one of use_patient, use_phase, use_slice_pos must be True"
        self.use_patient = use_patient
        self.use_phase = use_phase
        self.use_slice_pos = use_slice_pos
        self.debug = debug
        self.group_size = 1 + int(self.use_patient) + int(self.use_phase) + int(self.use_slice_pos)
        print(f"Debug custom loss with use_patient {self.use_patient}, use_phase {self.use_phase}, use_slice_pos {self.use_slice_pos}")
        super().__init__(**kwargs)
        print(f"Done setting up custom loss with use_patient {self.use_patient}, use_phase {self.use_phase}, use_slice_pos {self.use_slice_pos}")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for main_index in range(batch_size // self.group_size):
            grouped_indices = [(main_index * self.group_size) + grouped_index for grouped_index in range(self.group_size)]
            grouped_indices_with_augs = grouped_indices + [batch_size + index for index in grouped_indices]
            for grouped_index1, grouped_index2 in list(combinations(grouped_indices_with_augs, 2)):
                mask[grouped_index1, grouped_index2] = 0
                mask[grouped_index2, grouped_index1] = 0
        if self.debug:
            print("mask ", mask)
        return mask


class NT_Xent_Group_Pos(NT_Xent_Group_Neg):

    def __init__(self, mask_pos=(), **kwargs):
        self.mask_pos = mask_pos
        super().__init__(**kwargs)

    def get_positive_mask(self):
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for main_index in range(self.batch_size // self.group_size):
            grouped_indices = [(main_index * self.group_size) + grouped_index for grouped_index in range(self.group_size) if grouped_index not in self.mask_pos]
            grouped_indices_with_augs = grouped_indices + [self.batch_size + index for index in grouped_indices]
            for grouped_index1, grouped_index2 in list(combinations(grouped_indices_with_augs, 2)):
                mask[grouped_index1, grouped_index2] = 1
                mask[grouped_index2, grouped_index1] = 1
        if self.debug:
            print("mask ", mask)
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # includes the positive pairs (but don't include main diagonal)
        positive_samples = sim[self.get_positive_mask()].reshape(-1, 1)
        # removes elements on main diagonal as well as off diagonal augmentation pairs
        negative_samples = sim[self.mask].reshape(N, -1)

        # basically a multi-class problem but for each sample
        # the correct label is the first index for all the samples
        labels = torch.zeros(N).to(positive_samples.device).long()

        # this is why positive samples goes first in the concat
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


losses = {"cl": ContrastiveLoss,
          "nt_xent": NT_Xent,
          "nt_xent_neg": NT_Xent_Group_Neg,
          "nt_xent_pos": NT_Xent_Group_Pos,}
