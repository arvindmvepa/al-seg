import sys


class ModelFactory:

    def __init__(self):
        pass

    @staticmethod
    def create_model(model_type, **model_kwargs):
        if model_type == "spml_mv":
            from active_learning.model.spml_model import SPMLwMajorityVote
            model = SPMLwMajorityVote(**model_kwargs)
        elif model_type == "spml_sm":
            from active_learning.model.spml_model import SPMLwSoftmax
            model = SPMLwSoftmax(**model_kwargs)
        elif model_type == "dmpls":
            sys.path.append("./wsl4mis/code")
            from active_learning.model.dpmls_model import DMPLSModel
            model = DMPLSModel(**model_kwargs)
        elif model_type == "db_dmpls":
            sys.path.append("./wsl4mis/code")
            from active_learning.model.dpmls_model import DeepBayesianDMPLSModel
            model = DeepBayesianDMPLSModel(**model_kwargs)
        elif model_type == "strong":
            sys.path.append("./wsl4mis/code")
            from active_learning.model.strongly_sup_model import StronglySupModel
            model = StronglySupModel(**model_kwargs)
        elif model_type == "dmpls_em":
            sys.path.append("./wsl4mis/code")
            from active_learning.model.dmpls_entropy_mini_model import DMPLSEntropyMiniModel
            model = DMPLSEntropyMiniModel(**model_kwargs)
        elif model_type == "db_dmpls_em":
            sys.path.append("./wsl4mis/code")
            from active_learning.model.dmpls_entropy_mini_model import DeepBayesianDMPLSEntropyMiniModel
            model = DeepBayesianDMPLSEntropyMiniModel(**model_kwargs)
        elif model_type == "dmpls_mshah":
            sys.path.append("./wsl4mis/code")
            from active_learning.model.dmpls_MumfordShah_loss_model import DMPLSMumfordShahLossModel
            model = DMPLSMumfordShahLossModel(**model_kwargs)
        elif model_type == "db_dmpls_mshah":
            sys.path.append("./wsl4mis/code")
            from active_learning.model.dmpls_MumfordShah_loss_model import DeepBayesianDMPLSMumfordShahLossModel
            model = DeepBayesianDMPLSMumfordShahLossModel(**model_kwargs)
        elif model_type == "dmpls_s2l":
            sys.path.append("./wsl4mis/code")
            from active_learning.model.dmpls_s2l_model import DMPLSS2LModel
            model = DMPLSS2LModel(**model_kwargs)
        elif model_type == "db_dmpls_s2l":
            sys.path.append("./wsl4mis/code")
            from active_learning.model.dmpls_s2l_model import DeepBayesianDMPLSS2LModel
            model = DeepBayesianDMPLSS2LModel(**model_kwargs)
        else:
            raise ValueError(f"There is no model_type {model_type}")
        return model