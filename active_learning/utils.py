import torch


def seg_entropy_score(class_seg_prob_tensor):
    entropy_arr = pixel_entropy_score(class_seg_prob_tensor)
    mean_entropy = torch.mean(entropy_arr)
    return mean_entropy


def pixel_entropy_score(class_seg_prob_tensor):
    entropy_arr = class_seg_prob_tensor * torch.log(class_seg_prob_tensor + 1e-10)
    entropy_arr = torch.nan_to_num(entropy_arr, nan=0.0)
    entropy_arr = -torch.sum(entropy_arr, dim=0)

    return entropy_arr