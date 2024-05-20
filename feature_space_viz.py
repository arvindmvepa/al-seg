import os
from torch.utils.data import DataLoader
import sys
import numpy as np
from tqdm import tqdm
sys.path.append("./wsl4mis/code")
from active_learning.data_geometry.dataset import DatasetWrapper
import torch
from active_learning.data_geometry.net import resnet18, resnet50
from active_learning.feature_model.contrastive_net import ContrastiveLearner
import torchvision.transforms as T
from active_learning.dataset.dataset_factory import DatasetFactory
from active_learning.dataset.data_params import data_params


def get_encoder(encoder='resnet18', projection_dim=64, in_chns=1, pretrained=True, gpus="cuda:0"):
    if encoder == 'resnet18':
        print("Using Resnet18 for feature extraction...")
        encoder = resnet18(pretrained=pretrained, inchans=in_chns)
    elif encoder == 'resnet50':
        print("Using Resnet50 for feature extraction...")
        encoder = resnet50(pretrained=pretrained, inchans=in_chns)
    else:
        raise ValueError(f"Unknown feature model {encoder}")
    encoder = ContrastiveLearner(encoder, projection_dim=projection_dim)
    encoder = encoder.to(gpus)
    return encoder


def get_all_train_files():
    orig_train_im_list_file = data_params['train_file']
    with open(orig_train_im_list_file, "r") as f:
        return sorted(f.read().splitlines())


def get_dataset(dataset="ACDC", ann_type="scribble"):
    data_params_ = data_params[dataset][ann_type]
    data_root = data_params[dataset][ann_type]["data_root"]
    all_train_im_files = get_all_train_files(data_params_)
    all_train_full_im_paths = [os.path.join(data_root, im_path) for im_path in all_train_im_files]
    dataset = DatasetFactory.get_dataset(dataset, all_train_im_files, all_train_full_im_paths, data_params_)
    return dataset


def get_model_features(cl_model_path, dataset, inf_batch_size=128, gpus="cuda:0"):
    image_data, mage_meta_data, image_labels_arr = dataset.get_data()
    dataset = DatasetWrapper(image_data, transform=T.ToTensor())
    data_loader = DataLoader(dataset, batch_size=inf_batch_size, shuffle=False, pin_memory=True)
    features = torch.tensor([]).to(gpus)
    encoder = get_encoder()
    if os.path.exists(cl_model_path):
        encoder.load_state_dict(torch.load(cl_model_path))
    encoder = encoder.eval()
    with torch.no_grad():
        for inputs in tqdm(data_loader):
            inputs = inputs.to(gpus)
            # flatten the feature map (but not the batch dim)
            features_batch = encoder(inputs).flatten(1)
            features = torch.cat((features, features_batch), 0)
        feat = features.detach().cpu().numpy()
        torch.cuda.empty_cache()
    return feat


def get_contrastive_features(cl_model_path):
    dataset = get_dataset()
    features = get_model_features(cl_model_path, dataset)
    return features


if __name__ == "__main__":
    cl_model_path = "/home/amvepa91/cl_feature_model_acdc_new_1.pt"
    save_path = os.path.basename(cl_model_path).replace(".pth", ".npy")
    features = get_contrastive_features(cl_model_path)
    features = features.astype(np.float16)
    print(features.shape)
    np.save(save_path, features)

