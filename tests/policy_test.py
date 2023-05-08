"""test random_split and ranked_split"""
import pytest
from active_learning.model.base_model import BaseModel
from active_learning.model_uncertainty.base_model_uncertainty import BaseModelUncertainty
from active_learning.policy.base_policy import RandomActiveLearningPolicy
from active_learning.policy.ranked_policy import RankedPolicy
import os


@pytest.fixture
def get_random_policy(mocker, file_keys, im_key, train_files_dict, rounds, ann_file):
    """
    Mock model and model_uncertainty and create random policy object
    """
    # mock BaseModel
    mock_base_model = mocker.Mock(spec=BaseModel)
    mock_base_model.all_train_files_dict = train_files_dict
    mock_base_model.file_keys = file_keys
    mock_base_model.im_key = im_key
    mocker.patch.multiple(BaseModel, __abstractmethods__=set(),
                          get_round_train_file_paths=mocker.Mock(return_value=ann_file))
    # mock BaseModelUncertainty
    mock_base_model_uncertainty = mocker.Mock(spec=BaseModelUncertainty)
    mocker.patch.multiple(BaseModelUncertainty, __abstractmethods__=set())
    # instantiate RandomActiveLearningPolicy with mocked objects
    policy = RandomActiveLearningPolicy(mock_base_model, 
                                        mock_base_model_uncertainty, 
                                        rounds=rounds)
    yield policy
    # remove test.txt
    if os.path.exists(ann_file):
        os.remove(ann_file)


@pytest.mark.parametrize("file_keys, im_key, train_files_dict, rounds, ann_file, exp_round1_count, ann_file", 
                         [(["images"], "images", {"images": [1, 2, 3, 4]}, [(0.25, 0.25)], "test.txt", 1)])
def test_random_split(get_random_policy, file_keys, im_key, train_files_dict, 
                      rounds, ann_file, exp_round1_count):
    """
    Test that random_split works correctly
    """
    random_policy = get_random_policy(file_keys, im_key, train_files_dict, rounds, ann_file)
    random_split_dict = random_policy.random_split()
    assert len(random_split_dict["images"]) == exp_round1_count


#def test_ranked_split(mocker):
#pass

