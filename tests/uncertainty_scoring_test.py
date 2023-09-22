"""test uncertainty functions"""
import pytest
import numpy as np
import torch
from active_learning.model_uncertainty.uncertainty_scoring_functions import (mean_score, 
                                                                             entropy_w_label_probs, 
                                                                             ensemble_variance_ratio)


@pytest.mark.parametrize("im_labels, expected_score",
                         # test for basic functionality - input array of shape (n_models, n_scores)
                         # corresponding to 4 models, 1 score
                         [(np.array([[.1],[.3],[.4],[.5]]), 0.325)
                          ])
def test_mean_score(im_labels, expected_score):
    """
    Test mean_score
    """
    im_labels = torch.from_numpy(im_labels)
    score = mean_score(im_labels)
    score = score.detach().cpu().numpy().item()

    assert np.allclose(score, expected_score)


@pytest.mark.parametrize("im_labels, expected_score",
                         # test for basic functionality - input array of shape (n_models, n_classes, height, width)
                         # corresponding to 4 models, 2 classes, 3x2 images
                         [(np.array([[[[.5, .3], [.5, .2], [.1, .4]], [[.2, .7], [.2, .8], [.3, .9]]],
                                     [[[.4, .5], [.3, .7], [.2, .3]], [[.2, .5], [.1, .2], [0, .9]]],
                                     [[[.7, .9], [.1, .6], [.05, .8]], [[.3, .4], [.2, .2], [.8, .1]]],
                                     [[[.1, .2], [.2, .3], [.2, .4]], [[.7, .3], [.8, .2], [.9, .1]]]]),
                           0.7008254828189039)
                          ])
def test_entropy_w_label_probs(im_labels, expected_score):
    """
    Test entropy_w_label_probs
    """
    im_labels = torch.from_numpy(im_labels)
    score = entropy_w_label_probs(im_labels)
    score = score.detach().cpu().numpy().item()

    assert np.allclose(score, expected_score)
    

@pytest.mark.parametrize("im_labels, expected_score",
                            # test for basic functionality - input array of shape (n_models, n_classes, height, width)
                            # corresponding to 4 models, 1 image, 2 classes, 3x2 image size 
                            [(np.array([[[[.5, .3], [.5, .2], [.1, .4]], [[.2, .7], [.2, .8], [.3, .9]]],
                                        [[[.4, .5], [.3, .7], [.2, .3]], [[.2, .5], [.1, .2], [0, .9]]],
                                        [[[.7, .9], [.1, .6], [.05, .8]], [[.3, .4], [.2, .2], [.8, .1]]],
                                        [[[.1, .2], [.2, .3], [.2, .4]], [[.7, .3], [.8, .2], [.9, .1]]]
                                        ]),
                                0.375)
                            ])
def test_ensemble_variance_ratio(im_labels, expected_score):
    im_labels = torch.from_numpy(im_labels)
    score = ensemble_variance_ratio(im_labels)
    score = score.detach().cpu().numpy().item()    
    assert np.allclose(score, expected_score)