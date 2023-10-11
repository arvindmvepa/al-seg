"""test calculate_representativeness for data_geometry classes"""
import pytest
from active_learning.data_geometry.data_geometry_factory import DataGeometryFactory
from active_learning.data_geometry.kcenter_greedy import kCenterGreedy
import os
from functools import partial


def mock_select_batch_(mocker_inst, already_selected, N, **kwargs):
    """remove the indices of the first N data that are not already selected"""
    ims_to_select = [i for i in range(len(mocker_inst.X)) if i not in already_selected][:N]
    return ims_to_select


@pytest.fixture
def data_geometry(mocker, data_geometry_type, geometry_kwargs, im_score_file, all_train_im_files):
    """
    Create data_geometry object for testing
    """
    data_geometry = DataGeometryFactory.create_data_geometry(data_geometry_type=data_geometry_type,
                                                             **geometry_kwargs)
    data_geometry.all_train_im_files = all_train_im_files

    # mock kCenterGreedy
    mock_kcenter_greedy_alg = mocker.Mock(spec=kCenterGreedy)
    mock_kcenter_greedy_alg.X = [i for i in range(len(all_train_im_files))]
    mock_kcenter_greedy_alg.select_batch_.side_effect = partial(mock_select_batch_, mock_kcenter_greedy_alg)

    # assign coreset alg to data_geometry
    data_geometry.basic_coreset_alg = mock_kcenter_greedy_alg

    yield data_geometry

    # remove score file created by test
    if os.path.exists(im_score_file):
        os.remove(im_score_file)


@pytest.mark.parametrize("data_geometry_type, geometry_kwargs, im_score_file, all_train_im_files, " \
                         "calculate_representativeness_kwargs, exp_output",
                         [
                           # standard test case
                           ("kcenter_greedy", {}, "scores.txt", ["1","2","3","4"],
                            {"im_score_file": "scores.txt", "num_samples": 2, "already_selected": ["2"]},
                            ["1", "3"]),
                           ])
def test_calculate_representativeness(data_geometry, calculate_representativeness_kwargs, exp_output):
    """
    Test that ranked_split works correctly
    """
    output = data_geometry.calculate_representativeness(**calculate_representativeness_kwargs)
    assert sorted(output) == sorted(exp_output)
