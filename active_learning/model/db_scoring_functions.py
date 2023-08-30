import torch
from active_learning.utils import seg_entropy_score


def pass_probs(outputs):
    return outputs


def entropy_score(outputs):
    avg_outputs = torch.mean(outputs, axis=0)
    entropy = seg_entropy_score(avg_outputs)
    return entropy


def bald_score(outputs):
    avg_outputs = torch.mean(outputs, axis=0)
    entropy_expected = -torch.sum(avg_outputs * torch.log(avg_outputs + 1e-10), dim=0)

    expected_entropy = torch.sum(torch.sum(outputs * torch.log(outputs + 1e-10), dim=0), dim=0) / outputs.shape[0]
    
    bald_score = entropy_expected + expected_entropy
    avg_bald_score = torch.mean(bald_score)
    return avg_bald_score


db_scoring_functions = {"pass_probs": pass_probs,
                        "entropy_score": entropy_score,
                        "bald_score": bald_score}