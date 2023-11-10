import torch 
import h5py
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
import numpy as np
from tqdm import tqdm 
from active_learning.data_geometry.base_adversarial import BaseAdversarial
from active_learning.data_geometry.vaal import VAE, Discriminator, AdversarySampler


class VAAL(BaseAdversarial):
    """Class for identifying representative data points using Coreset sampling. Based on https://github.com/sinhasam/vaal/blob/master/main.py#L57"""

    def __init__(self, vae_steps_per=2, discriminator_steps_per=1, beta=1, adversary_param=1, 
                 input_chann=1, vae_beta=1, **kwargs):
        super().__init__(**kwargs)
        self.vae_steps_per = vae_steps_per
        self.discriminator_steps_per = discriminator_steps_per
        self.beta = beta
        self.adversary_param = adversary_param
        self.input_chann = input_chann
        self.vae_beta = vae_beta
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def calculate_representativeness(self, im_score_file, num_samples, already_selected=[], skip=False, seed=42, **kwargs):
        if skip:
            print("Skipping Calculating VAAL!")
            return

        print("Calculating VAAL..")
        all_indices = np.arange(len(self.all_train_im_files))
        already_selected_indices = [self.all_train_im_files.index(i) for i in already_selected]
        labeled_sampler = sampler.SubsetRandomSampler(already_selected_indices)
        labeled_dataloader = DataLoader(self.pool,
                                        sampler=labeled_sampler,
                                        batch_size=self.batch_size,
                                        drop_last=False, pin_memory=True)
        unlabeled_indices = np.setdiff1d(all_indices, already_selected_indices)
        unlabeled_sampler = sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = DataLoader(self.pool,
                                          sampler=unlabeled_sampler,
                                          batch_size=self.batch_size,
                                          drop_last=False, pin_memory=True)
        
        vae = VAE(self.latent_dim, self.input_chann, self.gpus)
        discriminator = Discriminator(self.latent_dim)
        vae, discriminator = self.train_adv_model(vae, discriminator, unlabeled_dataloader, labeled_dataloader)
        vae_indices = self.sample_for_labeling(vae, discriminator, unlabeled_dataloader, num_samples)
        vae_indices = vae_indices.tolist()
        
        # write score file
        with open(im_score_file, "w") as f:
            # higher score for earlier added images
            scores = [score for score in range(len(vae_indices), 0, -1)]
            for i, im_file in enumerate(self.all_train_im_files):
                if i in vae_indices:
                    score = scores[vae_indices.index(i)]
                else:
                    score = 0
                f.write(f"{im_file},{score}\n")

        return [self.all_train_im_files[i] for i in vae_indices] 
        
        
    def train_adv_model(self, vae, discriminator, unlabeled_dataloader, labeled_dataloader):
        train_iterations = self.num_epochs * len(self.all_train_im_files) // self.batch_size
        labeled_data = self.read_data(labeled_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader)
        
        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
        
        vae.train()
        discriminator.train()
        vae = vae.to(self.gpus)
        discriminator = discriminator.to(self.gpus)
        
        for iter_count in tqdm(range(train_iterations)):
            labeled_imgs, _ = next(labeled_data)
            unlabeled_imgs, _ = next(unlabeled_data)
            labeled_imgs = labeled_imgs.to(self.gpus)
            unlabeled_imgs = unlabeled_imgs.to(self.gpus)
            
            # VAE step
            for count in range(self.vae_steps_per):
                recon, z, mu, logvar = vae(labeled_imgs)
                unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.vae_beta)
                unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
                transductive_loss = self.vae_loss(unlabeled_imgs, 
                        unlab_recon, unlab_mu, unlab_logvar, self.vae_beta)
            
                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)
                
                lab_real_preds = torch.ones(labeled_imgs.size(0), 1).to(self.gpus)
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0), 1).to(self.gpus)
                
                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                        self.bce_loss(unlabeled_preds, unlab_real_preds)
                total_vae_loss = unsup_loss + transductive_loss + self.adversary_param * dsc_loss
                optim_vae.zero_grad()
                total_vae_loss.backward()
                optim_vae.step()
                
                # sample new batch if needed to train the adversarial network
                if count < (self.vae_steps_per - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs, _ = next(unlabeled_data)
                    labeled_imgs = labeled_imgs.to(self.gpus)
                    unlabeled_imgs = unlabeled_imgs.to(self.gpus)
                    
            # Discriminator step
            for count in range(self.discriminator_steps_per):
                with torch.no_grad():
                    _, _, mu, _ = vae(labeled_imgs)
                    _, _, unlab_mu, _ = vae(unlabeled_imgs)
                
                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)
                
                lab_real_preds = torch.ones(labeled_imgs.size(0), 1).to(self.gpus)
                unlab_fake_preds = torch.ones(unlabeled_imgs.size(0), 1).to(self.gpus)
                
                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                           self.bce_loss(unlabeled_preds, unlab_fake_preds)

                optim_discriminator.zero_grad()
                dsc_loss.backward()
                optim_discriminator.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.discriminator_steps_per - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs, _ = next(unlabeled_data)
                    labeled_imgs = labeled_imgs.to(self.gpus)
                    unlabeled_imgs = unlabeled_imgs.to(self.gpus)
            
            if iter_count % 25 == 0:
                print(f'Iteration: {iter_count} | vae loss: {total_vae_loss.item()} | discriminator loss: {dsc_loss.item()}')
        
        return vae, discriminator
    
    @staticmethod
    def read_data(dataloader):
        while True:
            for img in dataloader:
                yield img
                
    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD
        
    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader, num_samples):
        sampler = AdversarySampler(num_samples)
        query_indices = sampler.sample(vae, discriminator, unlabeled_dataloader, self.gpus)
        return query_indices
        