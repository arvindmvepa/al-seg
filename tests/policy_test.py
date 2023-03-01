"""test random_split and ranked_split"""
import pytest
from active_learning.model.base_model import BaseModel
from active_learning.model_uncertainty.base_model_uncertainty import BaseModelUncertainty
from active_learning.policy.base_policy import RandomActiveLearningPolicy
from active_learning.policy.ranked_policy import RankedPolicy
import os


@pytest.fixture
def random_policy(mocker, file_keys, im_key, train_files_dict, split, ann_base_file):
    """
    Mock model and model_uncertainty and create random policy object
    """
    # mock BaseModel
    mocker.patch.multiple(BaseModel, __abstractmethods__=set())
    mock_base_model = mocker.Mock(spec=BaseModel)
    mock_base_model.all_train_files_dict = train_files_dict
    mock_base_model.file_keys = file_keys
    mock_base_model.im_key = im_key
    round_file_paths = {k: k+"_"+ann_base_file for k in file_keys}
    mock_base_model.get_round_train_file_paths.return_value = round_file_paths

    # mock BaseModelUncertainty
    mock_base_model_uncertainty = mocker.Mock(spec=BaseModelUncertainty)
    mocker.patch.multiple(BaseModelUncertainty, __abstractmethods__=set())

    # instantiate RandomActiveLearningPolicy with mocked objects and split percentage
    policy = RandomActiveLearningPolicy(mock_base_model, 
                                        mock_base_model_uncertainty)
    policy.cur_total_oracle_split = split

    yield policy

    # remove annotation files created by test
    for ann_file in round_file_paths.values():
        if os.path.exists(ann_file):
            os.remove(ann_file)


@pytest.fixture
def ranked_policy(mocker, file_keys, im_key, train_files_dict, im_score_file, im_score_dict, 
                  split, ann_base_file):
    """
    Mock model and model_uncertainty and create random policy object
    """
    # mock BaseModel
    mocker.patch.multiple(BaseModel, __abstractmethods__=set())
    mock_base_model = mocker.Mock(spec=BaseModel)
    mock_base_model.all_train_files_dict = train_files_dict
    mock_base_model.file_keys = file_keys
    mock_base_model.im_key = im_key
    round_file_paths = {k: k+"_"+ann_base_file for k in file_keys}
    mock_base_model.get_round_train_file_paths.return_value = round_file_paths

    # mock BaseModelUncertainty
    mock_base_model_uncertainty = mocker.Mock(spec=BaseModelUncertainty)
    mocker.patch.multiple(BaseModelUncertainty, __abstractmethods__=set())
    
    # instantiate RankedPolicy with mocked objects and split percentage
    policy = RankedPolicy(mock_base_model, 
                          mock_base_model_uncertainty)
    policy.cur_total_oracle_split = split
    policy.prev_round_im_score_file=im_score_file
    
    # create score file
    with open(im_score_file, "w") as f:
        for im, score in im_score_dict.items():
            f.write(f"{im},{score}\n")
            f.flush()

    yield policy

    # remove annotation files created by test
    for ann_file in round_file_paths.values():
        if os.path.exists(ann_file):
            os.remove(ann_file)
    # remove score file
    if os.path.exists(im_score_file):
        os.remove(im_score_file)


@pytest.mark.parametrize("file_keys, im_key, train_files_dict, split, " \
                         "ann_base_file, exp_split_count", 
                         [(["images"], "images", 
                           {"images": ["image1.png", "image2.png", "image3.png", "image4.png"]},
                           .25, "test.txt", 1)
                           ])
def test_random_split(random_policy, exp_split_count):
    """
    Test that random_split works correctly
    """
    random_split_dict = random_policy.random_split()
    assert len(random_split_dict[random_policy.model.im_key]) == exp_split_count


@pytest.mark.parametrize("file_keys, im_key, train_files_dict, im_score_file, im_score_dict, " \
                         "split, ann_base_file, exp_file_split", 
                         [
                           # standard test case
                           (["images"], "images", 
                           {"images": ["image1.png", "image2.png", "image3.png", "image4.png"]}, 
                           "scores.txt", 
                           {"image1.png": 0.1, "image2.png": 0.3, "image3.png": 0.2, "image4.png": 0.05},
                           .5, "test.txt", {"image2.png", "image3.png"}),
                           # test scientific notation
                           (["images"], "images", 
                           {"images": ["image1.png", "image2.png", "image3.png", "image4.png"]}, 
                           "scores.txt", 
                           {"image1.png": 0.1, "image2.png": 0.3, "image3.png": 0.2, "image4.png": 9.5e-5},
                           .5, "test.txt", {"image2.png", "image3.png"})
                           ])
def test_ranked_split(ranked_policy, exp_file_split):
    """
    Test that ranked_split works correctly
    """
    ranked_split_dict = ranked_policy.ranked_split()
    assert set(ranked_split_dict[ranked_policy.model.im_key]) == exp_file_split
