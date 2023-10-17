import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from active_learning.data_geometry.net import resnet18, resnet50
import torchvision.transforms as T
from active_learning.data_geometry.dataset import DatasetWrapper
from active_learning.data_geometry.contrastive_dataset import ContrastiveAugmentedDataSet, get_contrastive_augmentation
from active_learning.data_geometry.contrastive_net import ContrastiveLearner
from active_learning.data_geometry.contrastive_loss import losses


class FeatureModel:

    def __init__(self, model=None, patch_size=(256, 256), pretrained=True, model_ignore_layer=-1, inf_batch_size=128,
                 gpus="cuda:0"):
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.pretrained = pretrained
        self.ignore_layer = model_ignore_layer
        self.inf_batch_size = inf_batch_size
        self.gpus = gpus
        self.image_features = None

    def init_image_features(self, data):
        self.image_features = data
        dataset = DatasetWrapper(self.image_features, transform=T.ToTensor())
        data_loader = DataLoader(dataset, batch_size=self.inf_batch_size, shuffle=False, pin_memory=True)
        features = torch.tensor([]).to(self.gpus)
        model = self.get_model()
        model = model.eval()
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs.to(self.gpus)
                # convert from grayscale to color, hard-coded for pretrained resnet
                inputs = torch.cat([inputs, inputs, inputs], dim=1)
                # flatten the feature map (but not the batch dim)
                features_batch = model(inputs).flatten(1)
                features = torch.cat((features, features_batch), 0)
            feat = features.detach().cpu().numpy()
            torch.cuda.empty_cache()
        return feat

    def get_model(self):
        if self.model == 'resnet18':
            print("Using Resnet18 for feature extraction...")
            model = resnet18(pretrained=self.pretrained)
        elif self.model == 'resnet50':
            print("Using Resnet50 for feature extraction...")
            model = resnet50(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unknown feature model {self.model}")
        # only layers before feature_model_ignore_layer will be used for feature extraction
        model = nn.Sequential(*list(model.children())[:self.ignore_layer])
        model = model.to(self.gpus)
        return model

    def get_features(self):
        return self.image_features


class ContrastiveFeatureModel(FeatureModel):

    def __init__(self, lr=3e-4, batch_size=64, weight_decay=1.0e-6, temperature=0.5, projection_dim=64, num_epochs=100,
                 patch_size=(256,256), loss="nt_xent", **kwargs):
        super(ContrastiveFeatureModel).__init__(**kwargs)
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.projection_dim = projection_dim
        self.num_epochs = num_epochs
        self.patch_size = patch_size
        self.loss = loss

    def init_image_features(self, data):
        self.image_features = data

    def get_model(self):
        if self.model == 'resnet18':
            print("Using Resnet18 for feature extraction...")
            model = resnet18(pretrained=self.pretrained, inchans=1)
        elif self.model == 'resnet50':
            print("Using Resnet50 for feature extraction...")
            model = resnet50(pretrained=self.pretrained, inchans=1)
        else:
            raise ValueError(f"Unknown feature model {self.model}")
        # only layers before feature_model_ignore_layer will be used for feature extraction
        model = nn.Sequential(*list(model.children())[:self.ignore_layer])
        model = model.to(self.gpus)
        return model

    def get_features(self):
        dataset = DatasetWrapper(self.image_features, transform=T.ToTensor())
        data_loader = DataLoader(dataset, batch_size=self.inf_batch_size, shuffle=False, pin_memory=True)
        features = torch.tensor([]).to(self.gpus)
        model = self.get_model()
        self.train(model, self.image_features)
        model = model.eval()
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs.to(self.gpus)
                # flatten the feature map (but not the batch dim)
                features_batch = model(inputs).flatten(1)
                features = torch.cat((features, features_batch), 0)
            feat = features.detach().cpu().numpy()
            torch.cuda.empty_cache()
        return feat

    def train(self, model, data):
        print("Training feature model with contrastive loss...")
        model = ContrastiveLearner(model, projection_dim=self.projection_dim)
        model = model.train()

        contrastive_dataset = ContrastiveAugmentedDataSet(data, transform=get_contrastive_augmentation(
            patch_size=self.patch_size))
        contrastive_dataloader = DataLoader(contrastive_dataset, batch_size=self.batch_size, shuffle=True,
                                            pin_memory=True)
        criterion = losses[self.loss]
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        for epoch in range(self.num_epochs):
            loss_epoch = 0
            for step, (x_i, x_j) in enumerate(contrastive_dataloader):
                optimizer.zero_grad()
                x_i = x_i.device(self.gpus)
                x_j = x_j.device(self.gpus)

                z_i, z_j = model(x_i, x_j)

                loss = criterion(z_i, z_j)
                loss.backward()

                optimizer.step()

                print(f"Step [{step}/{len(contrastive_dataloader)}]\t Loss: {loss.item()}")

                loss_epoch += loss.item()
            print(f"Epoch {epoch} loss: {loss_epoch / len(contrastive_dataloader)}")
        print("Done training feature model with contrastive loss!")


class NoFeatureModel(FeatureModel):

    def __init__(self, **kwargs):
        super().__init__()

    def init_image_features(self, data):
        self.image_features = data.reshape(data.shape[0], -1)

    def get_model(self):
        return None

    def get_features(self):
        return self.image_features
