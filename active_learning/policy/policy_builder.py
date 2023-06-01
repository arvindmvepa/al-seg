from active_learning.model.model_factory import ModelFactory
from active_learning.model_uncertainty.model_uncertainty_factory import ModelUncertaintyFactory
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

    def with_model_params(self, model_params):
        self._model_params = model_params
        return self

    def with_model_uncertainty_params(self, model_uncertainty_params):
        self._model_uncertainty_params = model_uncertainty_params
        return self

    def with_policy_params(self, policy_params):
        self._policy_params = policy_params
        return self

    def build(self):
        self.check_params()

        model = ModelFactory.create_model(**self._model_params)
        model_uncertainty = ModelUncertaintyFactory.create_model_uncertainty(model=model,
                                                                             **self._model_uncertainty_params)
        return PolicyFactory.create_policy(model=model, model_uncertainty=model_uncertainty,
                                             exp_dir=self._exp_dir, **self._policy_params)

    def check_params(self, check_params=("_exp_dir", "_model_params", "_model_uncertainty_params", "_policy_params")):
        for params in check_params:
            assert hasattr(self, params), f"have to set {params}"

    @staticmethod
    def build_policy(exp_params_file):
        exp_params = load_yaml(exp_params_file)

        return PolicyBuilder()\
            .make_exp_dir(exp_params['exp_dir'], exp_params_file)\
            .with_model_params(exp_params['model'])\
            .with_model_uncertainty_params(exp_params['model_uncertainty'])\
            .with_policy_params(exp_params['policy'])\
            .build()


def load_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as stream:
        params = yaml.safe_load(stream)
    return params