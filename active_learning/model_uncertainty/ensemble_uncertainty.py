from active_learning.model_uncertainty.base_model_uncertainty import BaseModelUncertainty
from active_learning.model_uncertainty.uncertainty_scoring_functions import scoring_functions


class EnsembleUncertainty(BaseModelUncertainty):
    """Class for generating uncertainty based on an ensemble

    Attributes
    ----------
    score_func : func
        Scoring function used on ensemble predictions

    """


    def __init__(self, model, score_func="entropy_w_label_counts"):
        super().__init__(model=model)
        self.score_func=score_func

    def calculate_uncertainty(self, im_score_file, round_dir, ignore_ims_dict=None, skip=False, **kwargs):
        if skip:
            print("Skipping Calculating Uncertainty!")
            return
        print("Starting to Ensemble Predictions")
        self.model.get_ensemble_scores(score_func=scoring_functions[self.score_func], round_dir=round_dir,
                                       im_score_file=im_score_file, ignore_ims_dict=ignore_ims_dict, **kwargs)

