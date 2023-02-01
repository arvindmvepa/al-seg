from active_learning.model.model_factory import ModelFactory
from active_learning.model_uncertainty.model_uncertainty_factory import ModelUncertaintyFactory
from active_learning.policy.policy_factory import PolicyFactory
import yaml


class PolicyBuilder:

    def __init__(self):
        pass

    @staticmethod
    def build_policy(exp_params_file):
        exp_params = load_yaml(exp_params_file)
        model_params = exp_params['model']
        model = ModelFactory.create_model(**model_params)
        model_uncertainty_params = exp_params['model_uncertainty']
        model_uncertainty = ModelUncertaintyFactory.create_model_uncertainty(model=model, **model_uncertainty_params)
        policy_params = exp_params['policy']
        policy = PolicyFactory.create_policy(model=model, model_uncertainty=model_uncertainty,
                                             exp_params_file=exp_params_file, **policy_params)
        return policy


def load_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as stream:
        params = yaml.safe_load(stream)
    return params