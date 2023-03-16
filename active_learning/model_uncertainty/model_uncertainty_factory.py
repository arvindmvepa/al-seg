from active_learning.model_uncertainty.base_model_uncertainty import NoModelUncertainty
from active_learning.model_uncertainty.ensemble_uncertainty import EnsembleUncertainty


class ModelUncertaintyFactory:

    def __init__(self):
        pass

    @staticmethod
    def create_model_uncertainty(model_uncertainty_type, **model_uncertainty_kwargs):
        if model_uncertainty_type == "no":
            model_uncertainty = NoModelUncertainty(**model_uncertainty_kwargs)
        elif model_uncertainty_type == "ensemble":
            model_uncertainty = EnsembleUncertainty(**model_uncertainty_kwargs)
        else:
            raise ValueError(f"There is no model_uncertainty_type {model_uncertainty_type}")
        return model_uncertainty