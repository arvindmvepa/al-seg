import os
import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader, Sampler
from active_learning.data_geometry.net import resnet18, resnet50
import torchvision.transforms as T
from active_learning.data_geometry.dataset import DatasetWrapper
from active_learning.feature_model.contrastive_dataset import ContrastiveAugmentedDataSet, get_contrastive_augmentation
from active_learning.feature_model.contrastive_sampler import PatientPhaseSliceBatchSampler
from active_learning.feature_model.contrastive_net import ContrastiveLearner
from active_learning.feature_model.contrastive_loss import losses


class FeatureModel(object):

    def __init__(self, exp_dir=None, encoder='resnet18', patch_size=(256, 256), in_chns=1, pretrained=True,
                 inf_batch_size=128, fuse_image_data=False, fuse_image_data_size_prop=.10, gpus="cuda:0"):
        super().__init__()
        self.exp_dir = exp_dir
        self.encoder = encoder
        self.patch_size = patch_size
        self.in_chns = in_chns
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
            encoder = resnet18(pretrained=self.pretrained, inchans=self.in_chns)
        elif self.encoder == 'resnet50':
            print("Using Resnet50 for feature extraction...")
            encoder = resnet50(pretrained=self.pretrained, inchans=self.in_chns)
        else:
            raise ValueError(f"Unknown feature model {self.encoder}")
        encoder = encoder.to(self.gpus)
        return encoder

    def get_features(self):
        model_features = self.get_model_features()
        if self.fuse_image_data:
            return self.fuse_image_data_with_model_features(model_features)
        else:
            print("Returning model features without fusing image data...")
            print("Model Features Shape: ", model_features.shape)
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

    def fuse_reg_image_data_with_model_features(self, model_features):
        print("Fusing reg image data with model features...")
        print("Original Model Features Shape: ", model_features.shape)
        image_data_size = self.reg_images.shape[1]
        model_features_size = model_features.shape[1]
        num_model_features_repeats = int(self.fuse_image_data_size_prop * image_data_size/model_features_size)
        model_features = np.repeat(model_features, num_model_features_repeats, axis=1)
        self.image_data_starting_index = 0
        self.image_data_ending_index = self.reg_images.shape[1]
        self.model_feature_starting_index = self.image_data_ending_index
        self.model_feature_ending_index = self.model_feature_starting_index + model_features.shape[1]

        fused_data = np.hstack((self.reg_images, model_features))
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
                 num_epochs=100, patch_size=(256,256), loss="nt_xent", loss_wt=1.0, neg_loss=None, neg_loss_wt=0.1,
                 pos_loss1=None, pos_loss1_wt=0.1, pos_loss1_mask=(), pos_loss2=None, pos_loss2_wt=0.1,
                 pos_loss2_mask=(), pos_loss3=None, pos_loss3_wt=0.1, pos_loss3_mask=(),
                 patience=5, tol=.01, cl_model_save_name="cl_feature_model.pt", use_patient=False,
                 use_phase=False, use_slice_pos=False, reset_sampler_every_epoch=False, seed=0, debug=False,
                 reg_data_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.projection_dim = projection_dim
        self.num_epochs = num_epochs
        self.patch_size = patch_size
        self.loss = loss
        self.loss_wt = loss_wt
        self.neg_loss = neg_loss
        self.neg_loss_wt = neg_loss_wt
        self.pos_loss1 = pos_loss1
        self.pos_loss1_wt = pos_loss1_wt
        self.pos_loss1_mask = pos_loss1_mask
        self.pos_loss2 = pos_loss2
        self.pos_loss2_wt = pos_loss2_wt
        self.pos_loss2_mask = pos_loss2_mask
        self.pos_loss3 = pos_loss3
        self.pos_loss3_wt = pos_loss3_wt
        self.pos_loss3_mask = pos_loss3_mask
        if self.neg_loss or self.pos_loss1 or self.pos_loss2 or self.pos_loss3:
            self.extra_loss = True
        else:
            self.extra_loss = False
        self.patience = patience
        self.tol = tol
        self.cl_model_save_path = os.path.join(self.exp_dir, cl_model_save_name)
        self.use_patient = use_patient
        self.use_phase = use_phase
        self.use_slice_pos = use_slice_pos
        self.reset_sampler_every_epoch = reset_sampler_every_epoch
        self.reg_data_dir = reg_data_dir
        self.seed = seed
        self.debug = debug
        self._cfg_indices = None
        self._hierarchical_image_data = None
        self._hierarchical_flat_image_data = None
        self._hierarchical_flat_cfg_info_list = None

    def init_image_features(self, data, cfgs_arr, cfg_indices):
        self._cfg_indices = cfg_indices
        self._hierarchical_organizing(data, cfgs_arr)
        self._extract_reg_slices(data, cfgs_arr, cfg_indices)
        super().init_image_features(data, cfgs_arr, cfg_indices)

    def init_model_features(self):
        pass

    def get_encoder(self):
        if self.encoder == 'resnet18':
            print("Using Resnet18 for feature extraction...")
            encoder = resnet18(pretrained=self.pretrained, inchans=self.in_chns)
        elif self.encoder == 'resnet50':
            print("Using Resnet50 for feature extraction...")
            encoder = resnet50(pretrained=self.pretrained, inchans=self.in_chns)
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

    def get_reg_labels(self):
        return self.reg_labels

    def train(self, model):
        if os.path.exists(self.cl_model_save_path):
            model.load_state_dict(torch.load(self.cl_model_save_path))
            print("Loaded contrastive feature model from disk")
            return
        else:
            print("Unable to load contrastive feature model, training from scratch...")

        print("Training feature model with contrastive loss...")
        model = model.train()

        if self.extra_loss:
            print(f"Create custom sampler")
            sampler = PatientPhaseSliceBatchSampler(hierarchical_data=self._hierarchical_image_data,
                                                    flat_data=self._hierarchical_flat_image_data,
                                                    flat_cfg_info_data=self._hierarchical_flat_cfg_info_list,
                                                    batch_size=self.batch_size, seed=self.seed,
                                                    reset_every_epoch=self.reset_sampler_every_epoch,
                                                    use_patient=self.use_patient, use_phase=self.use_phase,
                                                    use_slice_pos=self.use_slice_pos, debug=self.debug,
                                                    shuffle=True)
            data = self._hierarchical_flat_image_data
        else:
            sampler = None
            data = self.image_data

        contrastive_dataset = ContrastiveAugmentedDataSet(data, transform=get_contrastive_augmentation(
            patch_size=self.patch_size))
        if self.extra_loss:
            contrastive_dataloader = DataLoader(contrastive_dataset, batch_sampler=sampler, pin_memory=True)
        else:
            contrastive_dataloader = DataLoader(contrastive_dataset, batch_size=self.batch_size,
                                                shuffle=True, drop_last=True, batch_sampler=sampler, pin_memory=True)
        criterion_list = []
        criterion = losses["nt_xent"](batch_size=self.batch_size, temperature=self.temperature)
        print("Create standard loss with loss wt: ", self.loss_wt)
        criterion_list.append((criterion, self.loss_wt))
        if self.neg_loss is not None:
            neg_criterion = losses["nt_xent_neg"](use_patient=self.use_patient, use_phase=self.use_phase,
                                                      use_slice_pos=self.use_slice_pos, batch_size=self.batch_size,
                                                      temperature=self.temperature, debug=self.debug)
            criterion_list.append((neg_criterion, self.neg_loss_wt))
            print("Create negative loss with loss wt: ", self.neg_loss_wt)
        if self.pos_loss1 is not None:
            pos_criterion1 = losses["nt_xent_pos"](use_patient=self.use_patient, use_phase=self.use_phase,
                                                   use_slice_pos=self.use_slice_pos, batch_size=self.batch_size,
                                                   temperature=self.temperature, mask_pos=self.pos_loss1_mask, debug=self.debug)
            criterion_list.append((pos_criterion1, self.pos_loss1_wt))
            print(f"Create positive loss1 with loss wt: {self.pos_loss1_wt} and mask: {self.pos_loss1_mask}")
        if self.pos_loss2 is not None:
            pos_criterion2 = losses["nt_xent_pos"](use_patient=self.use_patient, use_phase=self.use_phase,
                                                   use_slice_pos=self.use_slice_pos, batch_size=self.batch_size,
                                                   temperature=self.temperature, mask_pos=self.pos_loss2_mask, debug=self.debug)
            criterion_list.append((pos_criterion2, self.pos_loss2_wt))
            print(f"Create positive loss2 with loss wt: {self.pos_loss2_wt} and mask: {self.pos_loss2_mask}")
        if self.pos_loss3 is not None:
            pos_criterion3 = losses["nt_xent_pos"](use_patient=self.use_patient, use_phase=self.use_phase,
                                                   use_slice_pos=self.use_slice_pos, batch_size=self.batch_size,
                                                   temperature=self.temperature, mask_pos=self.pos_loss3_mask, debug=self.debug)
            criterion_list.append((pos_criterion3, self.pos_loss3_wt))
            print(f"Create positive loss3 with loss wt: {self.pos_loss3_wt} and mask: {self.pos_loss3_mask}")


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

                loss = 0
                for criterion, wt in criterion_list:
                    loss += wt * criterion(z_i, z_j)
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

    def _hierarchical_organizing(self, data, cfgs_arr):
        """Organize the data into a hierarchical list structure based on patient, phase, and slice position"""
        hierarchical_image_data_dict = {}
        for (datum, cfg_arr) in zip(data, cfgs_arr):
            patient_id = cfg_arr[self._cfg_indices['patient_starting_index']:self._cfg_indices['patient_ending_index']][0]
            patient_dict = hierarchical_image_data_dict.get(patient_id, {})
            group_id = cfg_arr[self._cfg_indices['phase_starting_index']:self._cfg_indices['phase_ending_index']][0]
            group_dict = patient_dict.get(group_id, {})
            slice_pos = cfg_arr[self._cfg_indices['slice_pos_starting_index']:self._cfg_indices['slice_pos_ending_index']][0]
            group_dict[slice_pos] = datum
            patient_dict[group_id] = group_dict
            hierarchical_image_data_dict[patient_id] = patient_dict
        hierarchical_image_data_list = []
        hierarchical_flat_image_data_list = []
        hierarchical_flat_cfg_info_list = []
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
                    hierarchical_flat_cfg_info_list.append({"patient_id": patient_id, "group_id": group_id, "slice_pos": slice_pos})
                patient_lst.append(group_lst)
            hierarchical_image_data_list.append(patient_lst)
        self._hierarchical_image_data = hierarchical_image_data_list
        self._hierarchical_flat_image_data = hierarchical_flat_image_data_list
        self._hierarchical_flat_cfg_info_list = hierarchical_flat_cfg_info_list


    def _extract_reg_slices(self, data, cfgs_arr, cfg_indices):
        """Extract slices from the data that are registered"""
        reg_images = []
        reg_labels = []
        for (datum, cfg_arr) in zip(data, cfgs_arr):
            patient_num = int(cfg_arr[cfg_indices['patient_starting_index']:cfg_indices['patient_ending_index']][0])
            reg_index = int(cfg_arr[cfg_indices['reg_starting_index']:cfg_indices['reg_ending_index']][0])
            patient_file = "patient" + str(patient_num).zfill(3) + ".h5"
            patient_file = os.path.join(self.reg_data_dir, patient_file)
            h5f = h5py.File(patient_file, 'r')
            image = h5f.get('reg_image').value
            label = h5f.get('reg_scribble').value
            image = image.squeeze().transpose(1, 2, 0)[:, reg_index].flatten()
            label = label[:, reg_index].flatten()
            reg_images.append(image)
            reg_labels.append(label)
        self.reg_images = np.array(reg_images)
        self.reg_labels = np.array(reg_labels)


class NoFeatureModel(FeatureModel):

    def __init__(self, **kwargs):
        super().__init__(encoder=None)

    def init_model_features(self):
        self.image_features = self.flat_image_data

    def get_encoder(self):
        return None
