"""test random_split and ranked_split"""
import pytest
from pytest_mock import Mock
from active_learning.model.base_model import BaseModel
from active_learning.model_uncertainty.base_model_uncertainty import BaseModelUncertainty
from active_learning.policy.base_policy import RandomActiveLearningPolicy
from active_learning.policy.ranked_policy import RankedActiveLearningPolicy
import os


@pytest.fixture
def random_policy(mocker):
    """
    Mock model and model_uncertainty and create random policy object
    """
    base_model_obj = mocker.Mock(spec=BaseModel)
    mock_get_round_train_file_paths = mocker.MagicMock()
    mock_get_round_train_file_paths.return_value = "test.txt"
    mocker.patch.multiple(BaseModel, __abstractmethods__=set(), get_round_train_file_paths=mock_get_round_train_file_paths)
    base_model_uncertainty_obj = mocker.Mock(spec=BaseModelUncertainty)
    mocker.patch.multiple(BaseModelUncertainty, __abstractmethods__=set())

    random_policy = RandomActiveLearningPolicy(base_model_obj, base_model_uncertainty_obj, rounds=[(0.3, 0.3, .4)])
    random_policy.all_train_files_dict = {"images": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    yield random_policy
    os.remove("test.txt")


def test_random_split(random_policy):
    """
    Test that random_split works correctly
    """
    random_split_dict = random_policy.random_split()

    assert len(random_split_dict["images"]) == 3


#def test_ranked_split(mocker):
#pass

