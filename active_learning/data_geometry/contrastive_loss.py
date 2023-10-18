import torch
import torch.nn as nn


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
    def __init__(self, batch_size, temperature=0.5):
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
        print("N: ", N)

        z = torch.cat((z_i, z_j), dim=0)
        print("z shape: ", z.shape)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        print("sim shape: ", sim.shape)

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        print("sim_i_j shape: ", sim_i_j.shape)
        print("sim_j_i shape: ", sim_j_i.shape)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        print("positive_samples shape: ", positive_samples.shape)
        print("negative_samples shape: ", negative_samples.shape)

        labels = torch.zeros(N).to(positive_samples.device).long()
        print("labels shape: ", labels.shape)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        print("logits shape: ", logits.shape)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


losses = {"cl": ContrastiveLoss,
          "nt_xent": NT_Xent}
