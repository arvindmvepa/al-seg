from active_learning.feature_model.feature_model import FeatureModel, NoFeatureModel, ContrastiveFeatureModel


class FeatureModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_feature_model(model, contrastive, **feature_model_kwargs):
        if not model:
            model = NoFeatureModel(**feature_model_kwargs)
        elif contrastive:
            model = ContrastiveFeatureModel(**feature_model_kwargs)
        else:
            model = FeatureModel(**feature_model_kwargs)
        return model
