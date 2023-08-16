import torch
from utils import seg_entropy_score


def mean_probs(outputs):
    avg_outputs = torch.mean(outputs, dim=0)
    print(f"avg_outputs.shape: {avg_outputs.shape}")
    return avg_outputs


def entropy_score(outputs):
    avg_outputs = torch.mean(outputs, dim=0)
    print(f"avg_outputs.shape: {avg_outputs.shape}")
    entropy = seg_entropy_score(avg_outputs)
    print(f"entropy.shape: {entropy.shape}")
    return entropy


db_scoring_functions = {"mean_probs": mean_probs,
                        "entropy_score": entropy_score}