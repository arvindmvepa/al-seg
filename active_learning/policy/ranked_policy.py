from active_learning.policy.base_policy import BaseActiveLearningPolicy
import os


class RankedPolicy(BaseActiveLearningPolicy):
    """Policy which splits data based on min/max a score function

    Attributes
    ----------
    save_im_score_file : str
        Base filename for score file
    cur_im_score_file : str
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

    def __init__(self, model, model_uncertainty=None, data_geometry=None, ensemble_kwargs=None, uncertainty_kwargs=None,
                 save_im_score_file="scores.txt", rank_type="desc", resume=False, rounds=(), exp_dir="test",
                 pseudolabels=False,
                 tag="", seed=0):
        super().__init__(model=model, model_uncertainty=model_uncertainty, ensemble_kwargs=ensemble_kwargs,
                         uncertainty_kwargs=uncertainty_kwargs, data_geometry=data_geometry,
                         rounds=rounds, resume=resume, pseudolabels=pseudolabels, exp_dir=exp_dir, tag=tag, seed=seed)
        self.save_im_score_file = save_im_score_file
        self.cur_im_score_file = None
        self.rank_type = rank_type

    def data_split(self):
        if self.current_round_split_method == "random":
            return self.random_split()
        elif self.cur_im_score_file:
            if not os.path.exists(self.cur_im_score_file):
                raise ValueError(f"Score file {self.cur_im_score_file} does not exist!")
            print("Using ranked split!")
            split_data = self.ranked_split()
            self.cur_im_score_file = None
            return split_data
        else:
            print("Using random split!")
            return self.random_split()

    def ranked_split(self):
        print("Splitting data using ranked scoring!")
        return self._data_split(self._ranked_sample_unann_indices)

    def _ranked_sample_unann_indices(self):
        unann_im_dict, num_samples = self._get_unann_train_file_paths(), self._get_unann_num_to_sample()
        # get the scores per image from the score file in this format: img_name, score
        im_scores_list = open(self.cur_im_score_file).readlines()
        im_scores_list = [im_score.strip().split(",") for im_score in im_scores_list]
        im_scores_list = [(im_score[0], float(im_score[1])) for im_score in im_scores_list]
        if self.rank_type == "desc":
            sorted_im_scores_list = sorted(im_scores_list, key=lambda x: x[1], reverse=True)
        elif self.rank_type == "inc":
            sorted_im_scores_list = sorted(im_scores_list, key=lambda x: x[1])
        else:
            raise ValueError(f"Behavior for rank_type {self.rank_type} is undefined")
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

