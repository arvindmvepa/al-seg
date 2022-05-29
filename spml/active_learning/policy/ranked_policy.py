from spml.active_learning.model.model_params import model_params
from spml.active_learning.utils.numpy_parallel import parallel_apply_along_axis
from spml.active_learning.policy.base_policy import BaseActiveLearningPolicy
import random
import os
import time


class RankedPolicy(BaseActiveLearningPolicy):
    """Policy which splits data based on min/max a score function

    Attributes
    ----------
    save_im_score_file : str
        Base filename for score file
    prev_round_im_score_file : str
        Full path for score file generated in previous round
    rank_type : str
        Rank scores based on "desc" or "asc."

    Methods
    -------
    data_split()
        Marks a subset of unannotated data to be labeled by the oracle.
        First round uses random_split() and subsequent rounds are split by
        ranked_split()
    ranked_split()
        Splits unannotated data to be labeled by the oracle based on rank
        from the score function in previous round.

    """

    def __init__(self, model, exp_params_file, model_uncertainty=None, ensemble_kwargs=None, uncertainty_kwargs=None,
                 save_im_score_file="scores.txt", rank_type="desc", rounds=(), exp_dir="test", pseudolabels=False,
                 tag="", seed=0):
        super().__init__(model=model, exp_params_file=exp_params_file, model_uncertainty=model_uncertainty,
                         ensemble_kwargs=ensemble_kwargs, uncertainty_kwargs=uncertainty_kwargs, rounds=rounds,
                         pseudolabels=pseudolabels, exp_dir=exp_dir, tag=tag, seed=seed)
        self.save_im_score_file = save_im_score_file
        self.prev_round_im_score_file = None
        self.rank_type = rank_type

    def _run_round(self):
        if self._round_num < (self.num_rounds - 1):
            save_im_score_file = os.path.join(self.round_dir, self.save_im_score_file)
            model_uncertainty_kwargs = {"im_score_file": save_im_score_file,
                                        "ignore_ims_dict": self.cur_oracle_ims,
                                        "round_dir": self.round_dir}
            self._run_round_models(calculate_model_uncertainty=True, uncertainty_kwargs=model_uncertainty_kwargs)
            self.prev_round_im_score_file = save_im_score_file
        else:
            self._run_round_models()

    def data_split(self):
        if self._round_num == 0:
            return self.random_split()
        else:
            return self.ranked_split()

    def ranked_split(self):
        print("Splitting data using ranked scoring!")
        return self._data_split(self._ranked_sample_unann_indices)

    def _ranked_sample_unann_indices(self):
        unann_im_dict, num_samples = self._get_unann_train_file_paths(), self._get_unann_num_samples()
        # get the scores per image from the score file
        im_scores_list = open(self.prev_round_im_score_file).readlines()
        im_scores_list = [im_score.strip().split(",") for im_score in im_scores_list]
        if self.rank_type == "desc":
            sorted_im_scores_list = sorted(im_scores_list, key=lambda x: x[1], reverse=True)
        elif self.rank_type == "inc":
            sorted_im_scores_list = sorted(im_scores_list, key=lambda x: x[1])
        else:
            raise ValueError(f"Behavior for rank_type {rank_type} is undefined")
        print(f"Top scores: {sorted_im_scores_list[:5]}")
        sorted_im_list = [im_score[0] for im_score in sorted_im_scores_list]

        # filter out training set images that we are using as our labeled set
        filtered_sorted_im_list = [top_im for top_im in sorted_im_list if
                                   all(top_im not in ann_im for ann_im in self.cur_oracle_ims[self.im_key])]
        top_im_list = filtered_sorted_im_list[:num_samples]

        # retrieve indices of unann files
        unann_ims = unann_im_dict[self.im_key]
        sampled_unann_indices = [i for i, unann_im in enumerate(unann_ims) for top_im in top_im_list if top_im in unann_im]
        return sampled_unann_indices