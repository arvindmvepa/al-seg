from abc import abstractmethod
import os
from tqdm import tqdm 
from scipy.ndimage.interpolation import zoom
import numpy as np 
import h5py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader
from active_learning.data_geometry.base_data_geometry import BaseDataGeometry

# TODO Add contrastive model
class BaseTypiclust(BaseDataGeometry):
    """Base class for Adversarial sampling"""

    def __init__(self, patch_size=(224, 224), gpus="cuda:0", 
                 use_model_features=False, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.gpus = gpus
        self.use_model_features = use_model_features
        
        # Load a pretrained ResNet-18 model
        self.embedding_model = models.resnet18(pretrained=True)
        # Remove the final fully connected layer (optional)
        self.embedding_model = torch.nn.Sequential(*(list(self.embedding_model.children())[:-1]))
        
    def setup(self, exp_dir, data_root, all_train_im_files):
        self.setup_data(data_root, all_train_im_files)

    def setup_data(self, data_root, all_train_im_files):
        print("Initializing Training pool X for Typiclust sampling!")
        self.data_root = data_root
        self.all_train_im_files = all_train_im_files
        self.all_train_full_im_paths = [os.path.join(data_root, im_path) for im_path in all_train_im_files]
        self.pool = DatasetWrapper(self.all_train_full_im_paths, transform=T.ToTensor(),
                                      patch_size=self.patch_size)
        
        all_imgs_dataloader = DataLoader(self.pool,
                                        batch_size=1,
                                        drop_last=False, 
                                        pin_memory=True, shuffle=False)
        self.features = self.get_semantic_embedding(all_imgs_dataloader)
        
        print("Done setting up data")

    def get_semantic_embedding(self, dataloader):
        arr = []
        # Generate embedding for all the images
        embedding_model = self.embedding_model.to(self.gpus)
        with torch.no_grad():
            print("Generating embeddings for all images..")
            for batch in tqdm(dataloader):
                # Convert to 3-channel image by repeating the first channel 3 times
                batch_3d = batch.repeat(1, 3, 1, 1)
                batch_3d = batch_3d.to(self.gpus)
                embeddings = embedding_model(batch_3d)
                embeddings = embeddings.view(embeddings.size(0), -1)
                arr.append(embeddings.cpu().numpy())
        return np.concatenate(arr, axis=0)
    
    @abstractmethod
    def select_samples(self):
        raise NotImplementedError()
    
    
class DatasetWrapper(Dataset):
    def __init__(self, all_train_full_im_paths, transform=None, patch_size=(256, 256)):
        self.sample_list = all_train_full_im_paths
        self.transform = transform
        self.patch_size = patch_size
        
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        case = self.sample_list[idx]
        with h5py.File(case, 'r') as hf:
            sample = hf['image'][:]
        sample = self._patch_im(sample, self.patch_size)
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    @staticmethod
    def _patch_im(im, patch_size):
        x, y = im.shape
        image = zoom(im, (patch_size[0] / x, patch_size[1] / y), order=0)
        return image