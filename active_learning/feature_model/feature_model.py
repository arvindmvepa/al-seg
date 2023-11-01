import os
import numpy as np
from numpy.random import RandomState
import torch
from torch.utils.data import DataLoader, Sampler
from active_learning.data_geometry.net import resnet18, resnet50
import torchvision.transforms as T
from active_learning.data_geometry.dataset import DatasetWrapper
from active_learning.feature_model.contrastive_dataset import ContrastiveAugmentedDataSet, get_contrastive_augmentation
from active_learning.feature_model.contrastive_net import ContrastiveLearner
from active_learning.feature_model.contrastive_loss import losses


class FeatureModel(object):

    def __init__(self, exp_dir=None, encoder='resnet18', patch_size=(256, 256), pretrained=True, inf_batch_size=128,
                 fuse_image_data=False, fuse_image_data_size_prop=.10, gpus="cuda:0"):
        super().__init__()
        self.exp_dir = exp_dir
        self.encoder = encoder
        self.patch_size = patch_size
        self.pretrained = pretrained
        self.inf_batch_size = inf_batch_size
        self.gpus = gpus
        self.fuse_image_data = fuse_image_data
        self.fuse_image_data_size_prop = fuse_image_data_size_prop
        self.model_feature_starting_index = None
        self.model_feature_ending_index = None
        self.image_data_starting_index = None
        self.image_data_ending_index = None
        self.image_data = None
        self.flat_image_data = None
        self.image_features = None

    def init_image_features(self, data, cfgs_arr, cfg_indices):
        self.image_data = data
        self.flat_image_data = self.image_data.reshape(self.image_data.shape[0], -1)
        self.init_model_features()

    def init_model_features(self):
        print("initializing image features from feature model...")
        dataset = DatasetWrapper(self.image_data, transform=T.ToTensor())
        data_loader = DataLoader(dataset, batch_size=self.inf_batch_size, shuffle=False, pin_memory=True)
        features = torch.tensor([]).to(self.gpus)
        encoder = self.get_encoder()
        encoder = encoder.eval()
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs.to(self.gpus)
                # convert from grayscale to color, hard-coded for pretrained resnet
                inputs = torch.cat([inputs, inputs, inputs], dim=1)
                # flatten the feature map (but not the batch dim)
                features_batch = encoder(inputs).flatten(1)
                features = torch.cat((features, features_batch), 0)
            feat = features.detach().cpu().numpy()
            torch.cuda.empty_cache()
        self.image_features = feat

    def get_encoder(self):
        if self.encoder == 'resnet18':
            print("Using Resnet18 for feature extraction...")
            encoder = resnet18(pretrained=self.pretrained)
        elif self.encoder == 'resnet50':
            print("Using Resnet50 for feature extraction...")
            encoder = resnet50(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unknown feature model {self.encoder}")
        encoder = encoder.to(self.gpus)
        return encoder

    def get_features(self):
        model_features = self.get_model_features()
        if self.fuse_image_data:
            return self.fuse_image_data_with_model_features(model_features)
        else:
            return model_features

    def fuse_image_data_with_model_features(self, model_features):
        print("Fusing image data with model features...")
        print("Original Model Features Shape: ", model_features.shape)
        image_data_size = self.flat_image_data.shape[1]
        model_features_size = model_features.shape[1]
        num_model_features_repeats = int(self.fuse_image_data_size_prop * image_data_size/model_features_size)
        model_features = np.repeat(model_features, num_model_features_repeats, axis=1)
        self.image_data_starting_index = 0
        self.image_data_ending_index = self.flat_image_data.shape[1]
        self.model_feature_starting_index = self.image_data_ending_index
        self.model_feature_ending_index = self.model_feature_starting_index + model_features.shape[1]

        fused_data = np.hstack((self.flat_image_data, model_features))
        print(f"Model features shape after repeats: {model_features.shape}")
        print(f"Fused features shape: {fused_data.shape}")
        return fused_data

    def get_model_features(self):
        return self.image_features

    def get_model_features_indices(self):
        return self.model_feature_starting_index, self.model_feature_ending_index

    def get_image_data_indices(self):
        return self.image_data_starting_index, self.image_data_ending_index


class ContrastiveFeatureModel(FeatureModel):

    def __init__(self, lr=3e-4, batch_size=64, weight_decay=1.0e-6, temperature=0.5, projection_dim=64,
                 num_epochs=100, patch_size=(256,256), loss="nt_xent", extra_loss=None, extra_loss_wt=0.1, patience=5,
                 tol=.01, cl_model_save_name="cl_feature_model.pt", use_patient=False, use_phase=False,
                 use_slice_pos=False, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.projection_dim = projection_dim
        self.num_epochs = num_epochs
        self.patch_size = patch_size
        self.loss = loss
        self.extra_loss = extra_loss
        self.extra_loss_wt = extra_loss_wt
        self.patience = patience
        self.tol = tol
        self.cl_model_save_path = os.path.join(self.exp_dir, cl_model_save_name)
        self.use_patient = use_patient
        self.use_phase = use_phase
        self.use_slice_pos = use_slice_pos
        self.seed = seed
        self._hierarchical_image_data = None
        self._hierarchical_flat_image_data = None

    def init_image_features(self, data, cfgs_arr, cfg_indices):
        self._hierarchical_organizing(data, cfgs_arr, cfg_indices)
        super().init_image_features(data, cfgs_arr, cfg_indices)

    def init_model_features(self):
        pass

    def get_encoder(self):
        if self.encoder == 'resnet18':
            print("Using Resnet18 for feature extraction...")
            encoder = resnet18(pretrained=self.pretrained, inchans=1)
        elif self.encoder == 'resnet50':
            print("Using Resnet50 for feature extraction...")
            encoder = resnet50(pretrained=self.pretrained, inchans=1)
        else:
            raise ValueError(f"Unknown feature model {self.encoder}")
        encoder = ContrastiveLearner(encoder, projection_dim=self.projection_dim)
        encoder = encoder.to(self.gpus)
        return encoder

    def get_model_features(self):
        dataset = DatasetWrapper(self.image_data, transform=T.ToTensor())
        data_loader = DataLoader(dataset, batch_size=self.inf_batch_size, shuffle=False, pin_memory=True)
        features = torch.tensor([]).to(self.gpus)
        encoder = self.get_encoder()
        self.train(encoder)
        encoder = encoder.eval()
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs.to(self.gpus)
                # flatten the feature map (but not the batch dim)
                features_batch = encoder(inputs).flatten(1)
                features = torch.cat((features, features_batch), 0)
            feat = features.detach().cpu().numpy()
            torch.cuda.empty_cache()
        return feat

    def train(self, model):
        if os.path.exists(self.cl_model_save_path):
            model.load_state_dict(torch.load(self.cl_model_save_path))
            print("Loaded contrastive feature model from disk")
            return
        else:
            print("Unable to load contrastive feature model, training from scratch...")

        print("Training feature model with contrastive loss...")
        model = model.train()

        if self.extra_loss is not None:
            print(f"Create custom sampler for extra loss {self.extra_loss}")
            sampler = PatientPhaseSliceBatchSampler(hierarchical_data=self._hierarchical_image_data,
                                                    flat_data=self._hierarchical_flat_image_data,
                                                    batch_size=self.batch_size, seed=self.seed,
                                                    use_patient=self.use_patient, use_phase=self.use_phase,
                                                    use_slice_pos=self.use_slice_pos)
            data = self._hierarchical_flat_image_data
        else:
            sampler = None
            data = self.image_data

        contrastive_dataset = ContrastiveAugmentedDataSet(data, transform=get_contrastive_augmentation(
            patch_size=self.patch_size))
        contrastive_dataloader = DataLoader(contrastive_dataset, batch_size=self.batch_size, shuffle=True,
                                            drop_last=True, sampler=sampler, pin_memory=True)
        criterion = losses[self.loss](batch_size=self.batch_size, temperature=self.temperature)
        if self.extra_loss is not None:
            extra_criterion = losses[self.extra_loss](use_patient=self.use_patient, use_phase=self.use_phase,
                                                      use_slice_pos=self.use_slice_pos, batch_size=self.batch_size,
                                                      temperature=self.temperature)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        min_loss = None
        wait_time = 0
        for epoch in range(self.num_epochs):
            loss_epoch = 0
            for step, (x_i, x_j) in enumerate(contrastive_dataloader):
                optimizer.zero_grad()
                x_i = x_i.to(self.gpus)
                x_j = x_j.to(self.gpus)

                z_i, z_j = model(x_i, x_j)

                loss = criterion(z_i, z_j)
                if self.extra_loss is not None:
                    extra_loss = extra_criterion(z_i, z_j)
                    loss += self.extra_loss_wt * extra_loss
                loss.backward()

                optimizer.step()

                print(f"Step [{step}/{len(contrastive_dataloader)}]\t Loss: {loss.item()}")

                loss_epoch += loss.item()
            print(f"Epoch {epoch} loss: {loss_epoch / len(contrastive_dataloader)}")
            if min_loss is None:
                min_loss = None
            elif (min_loss - loss_epoch) > self.tol:
                min_loss = loss_epoch
                wait_time = 0
            elif wait_time >= self.patience:
                print("Early stopping!")
                break
            else:
                wait_time += 1
        print("Done training feature model with contrastive loss!")
        torch.save(model.state_dict(), self.cl_model_save_path)
        print("Saved CL feature model!")

    def _hierarchical_organizing(self, data, cfgs_arr, cfg_indices):
        """Organize the data into a hierarchical list structure based on patient, phase, and slice position"""
        hierarchical_image_data_dict = {}
        for (datum, cfg_arr) in zip(data, cfgs_arr):
            patient_id = cfg_arr[cfg_indices['patient_starting_index']:cfg_indices['patient_ending_index']][0]
            patient_dict = hierarchical_image_data_dict.get(patient_id, {})
            group_id = cfg_arr[cfg_indices['phase_starting_index']:cfg_indices['phase_ending_index']][0]
            group_dict = patient_dict.get(group_id, {})
            slice_pos = cfg_arr[cfg_indices['slice_pos_starting_index']:cfg_indices['slice_pos_ending_index']][0]
            group_dict[slice_pos] = datum
            patient_dict[group_id] = group_dict
            hierarchical_image_data_dict[patient_id] = patient_dict
        hierarchical_image_data_list = []
        hierarchical_flat_image_data_list = []
        for patient_id in sorted(hierarchical_image_data_dict.keys()):
            patient_dict = hierarchical_image_data_dict[patient_id]
            patient_lst = []
            for group_id in sorted(patient_dict.keys()):
                group_dict = patient_dict[group_id]
                group_lst = []
                for slice_pos in sorted(group_dict.keys()):
                    slice = group_dict[slice_pos]
                    group_lst.append(slice)
                    hierarchical_flat_image_data_list.append(slice)
                patient_lst.append(group_lst)
            hierarchical_image_data_list.append(patient_lst)
        self._hierarchical_image_data = hierarchical_image_data_list
        self._hierarchical_flat_image_data = hierarchical_flat_image_data_list

class NoFeatureModel(FeatureModel):

    def __init__(self, **kwargs):
        super().__init__(encoder=None)

    def init_model_features(self):
        self.image_features = self.flat_image_data

    def get_encoder(self):
        return None


class PatientPhaseSliceBatchSampler(Sampler):

    def __init__(self, flat_data, hierarchical_data, batch_size, seed=0, use_patient=False, use_phase=False, use_slice_pos=False):
        """Data is stored in a hierarchical list structure based on patient, phase, and slice position"""
        self.flat_data = flat_data
        self.hierarchical_data = hierarchical_data
        self.batch_size = batch_size
        self.random_state = RandomState(seed=seed)
        self.use_patient = use_patient
        self.use_phase = use_phase
        self.use_slice_pos = use_slice_pos
        assert self.use_patient or self.use_phase or self.use_slice_pos, "Must use at least one of patient, phase, or slice position"
        self.num_positives_per_batch = 1 + int(self.use_patient) + int(self.use_phase) + int(self.use_slice_pos)
        assert self.batch_size % self.num_positives_per_batch == 0, "Batch size must be divisible by the number of positives per batch"
        assert self.batch_size > self.num_positives_per_batch, "Batch size must be greater than the number of positives per batch"
        self.batches = None
        self.setup()
        print(f"Done setting up custom batch sampler with {len(self.batches)} batches that uses {self.num_positives_per_batch} positives per batch and use_patient {self.use_patient}, use_phase {self.use_phase}, use_slice_pos {self.use_slice_pos}, data length of {len(self.flat_data)}, and total number of samples {len(self)}")

    def __len__(self):
        return len(self.flat_data) * self.num_positives_per_batch

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def setup(self):
        nested_by_patient_index_groups, flat_index_groups = self.generate_nested_and_flat_patient_data_groups()
        self.batches = []
        patient_group_meta_indices = np.arange(len(nested_by_patient_index_groups))
        while len(flat_index_groups) > 1 and len(patient_group_meta_indices) > 1:
            random_patient_index_group, random_patient_group_meta_index = self.get_valid_random_patient_index_group_and_meta_index(
                patient_group_meta_indices,
                nested_by_patient_index_groups)
            if random_patient_index_group is None:
                break
            random_patient_index_group_, _ = self.get_valid_random_patient_index_group_and_meta_index(
                patient_group_meta_indices,
                nested_by_patient_index_groups,
                non_matching_patient_group_meta_index=random_patient_group_meta_index)
            if random_patient_index_group_ is None:
                break

            random_slice_index_group = self.random_state.choice(random_patient_index_group)
            random_slice_index_group_ = self.random_state.choice(random_patient_index_group_)
            new_batch = random_slice_index_group + random_slice_index_group_
            self.batches.append(new_batch)

            random_patient_index_group.remove(random_slice_index_group)
            random_patient_index_group_.remove(random_slice_index_group_)


    def get_valid_random_patient_index_group_and_meta_index(self, patient_group_meta_indices, nested_by_patient_index_groups,
                                                            non_matching_patient_group_meta_index=None):
        if len(patient_group_meta_indices) == 0:
            return None, None
        if (non_matching_patient_group_meta_index in patient_group_meta_indices) and len(patient_group_meta_indices) == 1:
            return None, None
        random_patient_group_meta_index = self.random_state.choice(patient_group_meta_indices)
        if random_patient_group_meta_index == non_matching_patient_group_meta_index:
            return self.get_valid_random_patient_index_group_and_meta_index(patient_group_meta_indices, nested_by_patient_index_groups,
                                                                            non_matching_patient_group_meta_index)
        random_patient_index_group = nested_by_patient_index_groups[random_patient_group_meta_index]
        if len(random_patient_index_group) == 0:
            patient_group_meta_indices.remove(random_patient_group_meta_index)
            return self.get_valid_random_patient_index_group_and_meta_index(patient_group_meta_indices, nested_by_patient_index_groups,
                                                                            non_matching_patient_group_meta_index)
        else:
            return random_patient_index_group, random_patient_group_meta_index


    def generate_nested_and_flat_patient_data_groups(self):
        flat_index_groups = []
        nested_by_patient_index_groups = []
        current_flat_index = 0
        for patient in self.hierarchical_data:
            patient_data_groups = []
            phase_1 = patient[0]
            phase_2 = patient[1]
            full_slice_lst = phase_1 + phase_2
            for slice_list in [phase_1, phase_2]:
                for i, slice in enumerate(slice_list):
                    current_data_group = list()
                    current_data_group.append(current_flat_index)
                    if self.use_slice_pos:
                        if i == 0:
                            current_data_group.append(current_flat_index + 1)
                        elif i == len(slice_list) - 1:
                            current_data_group.append(current_flat_index - 1)
                        else:
                            # randomly pick a slice before or after the current slice
                            random_slice_offset = self.random_state.choice([-1, 1])
                            current_data_group.append(current_flat_index + random_slice_offset)
                    if self.use_phase:
                        # randomly pick a slice in the same patient and phase
                        current_phase_flat_index = current_flat_index - i
                        random_slice_in_phase = None
                        while random_slice_in_phase not in current_data_group:
                            random_slice_in_phase = current_phase_flat_index + self.random_state.choice(np.arange(len(slice_list)))
                        current_data_group.append(random_slice_in_phase)
                    if self.use_patient:
                        # randomly pick a slice in the same patient
                        if slice_list is phase_1:
                            current_patient_flat_index = current_flat_index - i
                        else:
                            current_patient_flat_index = current_flat_index - i - len(phase_1)
                        random_slice_in_patient = None
                        while random_slice_in_patient not in current_data_group:
                            random_slice_in_patient = current_patient_flat_index + self.random_state.choice(np.arange(len(full_slice_lst)))
                    current_data_group.append(random_slice_in_patient)
                    patient_data_groups.append(current_data_group)
                    flat_index_groups.append(current_data_group)
                    current_flat_index += 1
            nested_by_patient_index_groups.append(patient_data_groups)
        return nested_by_patient_index_groups, flat_index_groups