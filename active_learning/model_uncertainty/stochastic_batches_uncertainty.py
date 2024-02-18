from random import sample
import numpy as np
from active_learning.model_uncertainty.ensemble_uncertainty import EnsembleUncertainty
from active_learning.model_uncertainty.uncertainty_scoring_functions import scoring_functions


class StochasticBatchesUncertainty(EnsembleUncertainty):
    """Class for generating uncertainty based on an stochastic batches

    Attributes
    ----------
    score_func : func
        Scoring function used on ensemble predictions

    """


    def __init__(self, sbatch_size=10, **kwargs):
        super().__init__(**kwargs)
        self.sbatch_size=sbatch_size

    def calculate_uncertainty(self, im_score_file, round_dir, ignore_ims_dict=None, skip=False, **kwargs):
        if skip:
            print("Skipping Calculating Uncertainty!")
            return
        print("Starting to Ensemble Predictions")
        self.model.get_ensemble_scores(score_func=scoring_functions[self.score_func], round_dir=round_dir,
                                       im_score_file=im_score_file, ignore_ims_dict=ignore_ims_dict, **kwargs)
        print("Generating Stochastic Batches")
        with open(im_score_file, "r") as f:
            im_scores_list = f.readlines()

        # get the scores per image from the score file in this format: img_name, score
        im_scores_list = [im_score.strip().split(",") for im_score in im_scores_list]
        im_scores_list = [(im_score[0], float(im_score[1])) for im_score in im_scores_list]

        unann_im_files = [im_score for im_score in im_scores_list if im_score[0] not in ignore_ims_dict[self.model.im_key]]
        shuffled_unann_im_files = sample(unann_im_files, len(unann_im_files))

        # Calculate the number of full groups and the remainder
        num_groups, remainder = divmod(len(shuffled_unann_im_files), self.sbatch_size)
        if remainder > 0:
            num_groups += 1

        # Initialize the group list
        groups = []

        # Extract full groups
        for i in range(num_groups):
            start_index = i * self.sbatch_size
            end_index = start_index + self.sbatch_size
            groups.append(shuffled_unann_im_files[start_index:end_index])

        # calculate group scores
        group_scores = []
        for group in groups:
            group_score = np.mean([im_score[1] for im_score in group])
            group_scores.append((group, group_score))

        sorted_group_scores_list = sorted(group_scores, key=lambda x: x[1])
        flattened_scores_list = []
        for group_and_score in sorted_group_scores_list:
            group, group_score = group_and_score
            for im_file, _ in group:
                flattened_scores_list.append((im_file, group_score))
        print(flattened_scores_list)
        """
        with open(im_score_file, "w") as f:
            for im_file, score in flattened_scores_list:
                f.write(f"{im_file},{np.round(score, 7)}\n")
                f.flush()
        """



