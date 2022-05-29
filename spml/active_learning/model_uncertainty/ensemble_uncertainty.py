from spml.active_learning.model_uncertainty.base_model_uncertainty import BaseModelUncertainty
from spml.active_learning.model_uncertainty.utils import scoring_functions


class EnsembleUncertainty(BaseModelUncertainty):
    """Class for generating uncertainty based on an ensemble

    Attributes
    ----------
    score_func : func
        Scoring function used on ensemble predictions

    """


    def __init__(self, model, scoring_method="entropy_w_label_counts"):
        super().__init__(model=model)
        self.score_func=scoring_functions[scoring_method]

    def calculate_uncertainty(self, im_score_file, ignore_ims_dict, round_dir, skip=False, **kwargs):
        if skip:
            print("Skipping Calculating Uncertainty!")
            return
        self.model.get_ensemble_scores(score_func=self.score_func, round_dir=round_dir, im_score_file=im_score_file,
                                       ignore_ims_dict=ignore_ims_dict, **kwargs)

