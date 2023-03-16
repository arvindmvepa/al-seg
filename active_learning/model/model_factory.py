from active_learning.model.spml_model import SPMLModel
from active_learning.model.wsl4mis_model import DMPLSModel
import sys

class ModelFactory:

    def __init__(self):
        pass

    @staticmethod
    def create_model(model_type, **model_kwargs):
        if model_type == "spml":
            model = SPMLModel(**model_kwargs)
        elif model_type == "dmpls":
            sys.path.append("./wsl4mis/code")
            print(sys.path)
            model = DMPLSModel(**model_kwargs)
        else:
            raise ValueError(f"There is no model_type {model_type}")
        return model