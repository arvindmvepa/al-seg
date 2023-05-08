"""test random_split and ranked_split"""
import pytest
from active_learning.model.base_model import BaseModel
from active_learning.model_uncertainty.base_model_uncertainty import BaseModelUncertainty
from active_learning.policy.base_policy import RandomActiveLearningPolicy
from active_learning.policy.ranked_policy import RankedPolicy
import os


@pytest.fixture
def random_policy(mocker, train_files_dict, rounds):
    """
    Mock model and model_uncertainty and create random policy object
    """
    # mock BaseModel
    mocker.patch.object(BaseModel, "all_train_files_dict", train_files_dict)
    mocker.patch.multiple(BaseModel, __abstractmethods__=set(), 
                          get_round_train_file_paths=mocker.Mock(return_value = "test.txt"))
    base_model_obj = mocker.Mock(spec=BaseModel)
    # mock BaseModelUncertainty
    base_model_uncertainty_obj = mocker.Mock(spec=BaseModelUncertainty)
    mocker.patch.multiple(BaseModelUncertainty, __abstractmethods__=set())
    # instantiate RandomActiveLearningPolicy with mocked objects
    policy = RandomActiveLearningPolicy(base_model_obj, 
                                        base_model_uncertainty_obj, 
                                        rounds=rounds)
    yield policy
    # remove test.txt
    os.remove("test.txt")


@pytest.mark.parametrize("train_files_dict, rounds, exp_round1_count", [({"images": [1, 2, 3, 4]},
                                                                         [(0.25, 0.25)],
                                                                          1)])
def test_random_split(random_policy, train_files_dict, rounds, exp_round1_count):
    """
    Test that random_split works correctly
    """
    random_split_dict = random_policy(train_files_dict, rounds).random_split()
    assert len(random_split_dict["images"]) == exp_round1_count


#def test_ranked_split(mocker):
#pass

