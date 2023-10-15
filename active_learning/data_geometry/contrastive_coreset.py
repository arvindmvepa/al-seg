import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from active_learning.data_geometry.base_coreset import BaseCoreset
from active_learning.data_geometry.CL_utils import (ContrastiveLearner,
                                                    ContrastiveLoss, 
                                                    ContrastiveAugmentedDataSet,
                                                    ClassifierDataSet,
                                                    get_contrastive_augmentation)


class ContrastiveCoreset(BaseCoreset):
    """Class for identifying representative data points using Coreset sampling"""
    # TODO margin should be carefully chosen thru experiments
    def __init__(self, subset_size="all", num_epochs_cl=200, in_chns=1, batch_size=64, margin=.5, patch_size=(256, 256), 
                 starting_sample=5, loss_weight=.3, lr_cl=1e-3, wdecay=5e-4, num_classes=1, 
                 encoder_out_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.subset_size = subset_size
        self.num_epochs_cl = num_epochs_cl
        self.in_chns = in_chns
        self.batch_size = batch_size
        self.margin = margin
        self.patch_size = patch_size
        self.starting_sample = starting_sample
        self.loss_weight = loss_weight
        self.lr_cl = lr_cl
        self.wdecay = wdecay
        self.num_classes = num_classes
        self.encoder_out_dim = encoder_out_dim
        
    def calculate_representativeness(self, im_score_file, num_samples, prev_round_dir, train_logits_path,
                                     already_selected=[], skip=False, delete_preds=True, **kwargs):
        if skip:
            print("Skipping Calculating Coreset!")
            return
        # TODO Include pre-trained backbone
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
            all_indices = np.arange(len(self.all_train_im_files))
            already_selected_indices = [self.all_train_im_files.index(i) for i in already_selected]
            unlabeled_indices = np.setdiff1d(all_indices, already_selected_indices)
            if self.subset_size == "all":
                subset_size = len(unlabeled_indices)
            else:
                subset_size = self.subset_size
            assert subset_size <= len(unlabeled_indices), "subset_size must be less than the number of unlabeled indices"
            subset = self.random_state.choice(unlabeled_indices, subset_size, replace=False).tolist()
            # For contrastive learning, we need to use the 2D features for the conv layers
            # all_set: nparray: (N, 256, 256)
            all_set = self._get_data()                # For contrastive learning
            l_set = all_set[already_selected_indices] # For classification task
            u_set = all_set[subset]                   # For classification task
            
            contrastive_dataset = ContrastiveAugmentedDataSet(all_set, transform=get_contrastive_augmentation(patch_size=self.patch_size))
            contrastive_dataloader = DataLoader(contrastive_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            
            classifier_dataset = ClassifierDataSet(l_set, u_set)
            classifier_dataloader = DataLoader(classifier_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            
            contrastive_learner = ContrastiveLearner(self.in_chns, self.num_classes, self.encoder_out_dim).to(self.gpus)
            contrastive_loss = ContrastiveLoss(self.margin)
            classifier_loss = nn.BCEWithLogitsLoss()
            
            optim_backbone = optim.Adam(contrastive_learner.parameters(), lr=self.lr_cl, weight_decay=self.wdecay)
            optimizer = {'cl': optim_backbone}
            
            print("Start training contrastive learner..")
            
            contrastive_learner.train()
            
            for epoch in range(self.num_epochs_cl):
                for (images1, images_aug1, images_aug2), (images2, labels) in zip(contrastive_dataloader, classifier_dataloader):
                    optimizer['cl'].zero_grad()
                    
                    images1, images_aug1, images_aug2 = images1.to(self.gpus), images_aug1.to(self.gpus), images_aug2.to(self.gpus)
                    images2, labels = images2.to(self.gpus), labels.to(self.gpus)
                    
                    # Forward pass for images and their augmentations
                    contrast_output = contrastive_learner(images1, task="contrastive")
                    output_aug1 = contrastive_learner(images_aug1, task="contrastive")
                    output_aug2 = contrastive_learner(images_aug2, task="contrastive")

                    # Positive loss (between image and its augmentation)
                    loss_positive1 = contrastive_loss(contrast_output, output_aug1, torch.ones(contrast_output.shape[0]).to(self.gpus))
                    loss_positive2 = contrastive_loss(contrast_output, output_aug2, torch.ones(contrast_output.shape[0]).to(self.gpus))
                    
                    # Negative loss (between image and augmentations of other images)
                    labels_negative = torch.zeros(contrast_output.shape[0]).to(self.gpus)
                    loss_negative = contrastive_loss(contrast_output, torch.cat((output_aug1[-1:], output_aug1[:-1]), 0), labels_negative)
                    
                    loss_positive = (loss_positive1 + loss_positive2) / 2.0
                    contrast_loss = (loss_positive + loss_negative) / 2.0
                    
                    # Classification loss
                    class_outputs = contrastive_learner(images2, task="classification").squeeze(1)
                    class_loss = classifier_loss(class_outputs, labels)
                    
                    total_loss = self.loss_weight * contrast_loss + (1 - self.loss_weight) * class_loss
                    total_loss.backward()
                    optimizer['cl'].step()
                
                if epoch % 2 == 0:
                  print("CL, Epoch: ", epoch, "Loss: ", total_loss.item())
            
            contrastive_learner.eval()
            print("Getting feature representations for all unlabeled data..")
            with torch.no_grad():
                inputs = torch.from_numpy(all_set).float().to(self.gpus)           
                feat = contrastive_learner.extract_features(inputs)
                feat = feat.detach().cpu().numpy()
                coreset_inst = self.create_coreset_inst(feat)
                sample_indices += coreset_inst.select_batch_(already_selected_indices, num_samples)
                
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