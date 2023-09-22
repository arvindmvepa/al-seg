"""test uncertainty functions"""
import pytest
import numpy as np
import torch
from active_learning.model.db_scoring_functions import pass_probs, entropy_score


@pytest.mark.parametrize("im_labels, expected_score",
                         # test for basic functionality - input array of shape (T, n_classes, height, width)
                         # corresponding to 1 iteration, 2 classes, 3x2 images
                         [(np.array([[[[.5, .3], [.5, .2], [.1, .4]], [[.2, .7], [.2, .8], [.3, .9]]]]),
                           np.array([[[[.5, .3], [.5, .2], [.1, .4]], [[.2, .7], [.2, .8], [.3, .9]]]]))
                          ])
def test_pass_probs(im_labels, expected_score):
    """
    Test pass_probs
    """
    im_labels = torch.from_numpy(im_labels)
    score = pass_probs(im_labels)
    score = score.detach().cpu().numpy()

    assert np.allclose(score, expected_score)


@pytest.mark.parametrize("outputs, expected_score",
                         # test for basic functionality - input array of shape (T, n_classes, height, width)
                         # corresponding to 1 iteration, 2 classes, 3x2 images
                         [(np.array([[[[.5, .3], [.5, .2], [.1, .4]], [[.2, .7], [.2, .8], [.3, .9]]]]),
                           0.583496696227593)
                          ])
def test_entropy_score(outputs, expected_score):
    """
    Test entropy_score
    """
    outputs = torch.from_numpy(outputs)
    score = entropy_score(outputs)
    score = score.detach().cpu().numpy().item()

    assert np.allclose(score, expected_score)