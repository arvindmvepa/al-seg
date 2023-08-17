import torch
from active_learning.utils import seg_entropy_score


def mean_probs(outputs):
    print(f"outputs.shape: {outputs.shape}")
    avg_outputs = torch.mean(outputs, dim=0)
    print(f"avg_outputs.shape: {avg_outputs.shape}")
    return avg_outputs


def entropy_score(outputs):
    avg_outputs = torch.mean(outputs, dim=0)
    entropy = seg_entropy_score(avg_outputs)
    return entropy


db_scoring_functions = {"mean_probs": mean_probs,
                        "entropy_score": entropy_score}