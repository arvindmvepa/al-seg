import numpy as np
from PIL import Image
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
from torch.utils.data import Dataset


# Feature extraction layers
class Encoder(nn.Module):
    def __init__(self, in_chns=1, out_dim=128):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_chns, 64, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc = nn.Linear(128 * 64 * 64, out_dim)
        
    def forward(self, x):
        features = self.conv1(x)
        features = self.act1(features)
        features = self.maxpool1(features)
        
        features = self.conv2(features)
        features = self.act2(features)
        features = self.maxpool2(features)
        
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features        
    

# Projection Head for Contrastive Learning
class ProjectionHead(nn.Module):
    """Head for contrastive learning"""
    def __init__(self, input_dim=128, output_dim=64):
        super(ProjectionHead, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


# Projection head for binary classification
class ClassifierHead(nn.Module):
    """Head for binary classification"""
    def __init__(self, input_dim=128, num_classes=1):
        super(ClassifierHead, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    
# Combined Model
class ContrastiveLearner(nn.Module):
    """Combined model for contrastive learning and classification"""
    def __init__(self, in_chns=1, num_classes=1, encoder_out_dim=128):
        super(ContrastiveLearner, self).__init__()
        self.encoder = Encoder(in_chns, encoder_out_dim)
        self.projection_head = ProjectionHead(encoder_out_dim, 64)
        self.classifier_head = ClassifierHead(encoder_out_dim, num_classes)

    def forward(self, x, task="contrastive"):
        features = self.encoder(x)
        if task == "contrastive":
            return self.projection_head(features)
        elif task == "classification":
            return self.classifier_head(features)
        
    def extract_features(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return features
    
    
class ContrastiveLoss(nn.Module):
    """Contrastive loss: https://ieeexplore.ieee.org/abstract/document/1640964"""
    # TODO: Implement other contrastive loss functions
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - 
                                                                      euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
# TODO BUG: torchvision's transforms assume the input array is in range [0, 255],
# but the raw image data was normalized with min and max method, thus, simply multiplying
# by 255 will not work.
def get_contrastive_augmentation(patch_size=(256, 256)):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # Randomly apply some color jitter
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),

        # Randomly apply affine transformations: rotation, translation, scale, shear
        transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),

        # Random horizontal flip
        transforms.RandomHorizontalFlip(p=0.5),

        # Randomly crop the image and then resize it back to the patch_size
        transforms.RandomResizedCrop(patch_size[0], scale=(0.8, 1.0)),

        # Randomly apply grayscale with a probability
        transforms.RandomGrayscale(p=0.2),

        # Gaussian blur
        transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 2.0)),

        # Convert the PIL image to a PyTorch tensor
        transforms.ToTensor(),
        
        # Random erasing (also known as cutout)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
    ])
    return transform

        
class ContrastiveAugmentedDataSet(Dataset):
    def __init__(self, data_arr: np.array, transform):
        self.transform = transform 
        self.data_arr = data_arr
        
    def __len__(self):
        return self.data_arr.shape[0]
    
    def __getitem__(self, idx):
        img = self.data_arr[idx]
        img = torch.tensor(img, dtype=torch.float32)
        # Generate two different augmentations for the same image
        aug_img1, aug_img2 = self.transform(img), self.transform(img)
        return img.unsqueeze(0), aug_img1, aug_img2
    
    
class ClassifierDataSet(Dataset):
    def __init__(self, l_set: np.array, u_set: np.array):
        self.l_set = l_set
        self.u_set = u_set
        self.all_set = np.concatenate((self.l_set, self.u_set), axis=0)
        self.labels = np.concatenate((np.ones(self.l_set.shape[0]), 
                                      np.zeros(self.u_set.shape[0])), axis=0)
        
    def __len__(self):
        return self.all_set.shape[0]
    
    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        features = torch.tensor(self.all_set[idx], dtype=torch.float32).unsqueeze(0)
        return features, label