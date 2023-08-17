import torch
from active_learning.utils import seg_entropy_score


def pass_probs(outputs):
    print(f"outputs.shape: {outputs.shape}")
    return outputs


def entropy_score(outputs):
    entropy = seg_entropy_score(outputs)
    return entropy


db_scoring_functions = {"pass_probs": pass_probs,
                        "entropy_score": entropy_score}