import torch


def mean_probs(outputs):
    avg_outputs = torch.mean(outputs, dim=0)
    print(f"avg_outputs.shape: {avg_outputs.shape}")
    return avg_outputs


def bald_score(outputs, T):
    avg_outputs = torch.mean(outputs, dim=0)
    first_term = -torch.sum(avg_outputs * torch.log(avg_outputs + 1e-10), dim=0)
    # Compute the second term of the BALD equation
    second_term = torch.sum(torch.sum(outputs * torch.log(outputs + 1e-10), dim=0), dim=0)/T
    # Compute the BALD score
    bald_score = first_term + second_term
    print(f"bald_score.shape: {bald_score.shape}")
    return bald_score


db_scoring_functions = {"mean_probs": mean_probs,
                        "bald_score": bald_score}