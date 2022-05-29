from spml.active_learning.model.spml_model import SPMLModel


class ModelFactory:

    def __init__(self):
        pass

    @staticmethod
    def create_model(model_type, **model_kwargs):
        if model_type == "spml":
            model = SPMLModel(**model_kwargs)
        else:
            raise ValueError(f"There is no model_type {model_type}")
        return model