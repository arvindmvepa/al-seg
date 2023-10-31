from active_learning.model.model_factory import ModelFactory
from active_learning.model_uncertainty.model_uncertainty_factory import ModelUncertaintyFactory
from active_learning.data_geometry.data_geometry_factory import DataGeometryFactory
from active_learning.policy.policy_factory import PolicyFactory
import yaml
import os
import shutil


class PolicyBuilder:

    def __init__(self):
        pass

    def make_exp_dir(self, exp_dir, exp_params_file):
        self._exp_dir = exp_dir
        if not os.path.exists(self._exp_dir):
            os.makedirs(self._exp_dir)
        # copy exp_params_file to experiment directory
        shutil.copyfile(exp_params_file, 
                        os.path.join(self._exp_dir, "exp.yml"))
        return self

    def load_params(self, params):
        if "model" in params:
            self._model_params = params["model"]
        if "policy" in params:
            self._policy_params = params["policy"]
        if "model_uncertainty" in params:
            self._model_uncertainty_params = params["model_uncertainty"]
        if "data_geometry" in params:
            self._data_geometry_params = params["data_geometry"]
        return self

    def build(self):
        self.check_params()
        model = ModelFactory.create_model(**self._model_params)
        model_uncertainty = None
        data_geometry = None
        if hasattr(self, "_model_uncertainty_params"):
            model_uncertainty = ModelUncertaintyFactory.create_model_uncertainty(model=model,
                                                                                 **self._model_uncertainty_params)
        if hasattr(self, "_data_geometry_params"):
            data_geometry = DataGeometryFactory.create_data_geometry(model_uncertainty=model_uncertainty,
                                                                     ann_type=model.ann_type,
                                                                     **self._data_geometry_params)
        return PolicyFactory.create_policy(model=model, model_uncertainty=model_uncertainty,
                                           data_geometry=data_geometry, exp_dir=self._exp_dir, **self._policy_params)

    def check_params(self, check_params=("_exp_dir", "_model_params", "_policy_params")
                     ):
        for params in check_params:
            assert hasattr(self, params), f"have to set {params}"

    @staticmethod
    def build_policy(exp_params_file):
        exp_params = load_yaml(exp_params_file)

        return PolicyBuilder()\
            .make_exp_dir(exp_params.get('exp_dir'), exp_params_file)\
            .load_params(exp_params)\
            .build()


def load_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as stream:
        params = yaml.safe_load(stream)
    return params