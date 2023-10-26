from active_learning.data_geometry.kcenter_greedy_coreset import KCenterGreedyCoreset


class ProbKCenterGreedyCoreset(KCenterGreedyCoreset):

    def __init__(self, num_coreset_rounds, **kwargs):
        super().__init__(**kwargs)
        self.num_coreset_rounds = num_coreset_rounds
        
    def calculate_representativeness(self, im_score_file, num_samples, prev_round_dir, train_logits_path,
                                     already_selected=[], skip=False, delete_preds=True, **kwargs):
        if skip:
            print("Skipping Calculating ProbKCenterGreedyCoreset!")
            return

        print("Calculating ProbKCenterGreedyCoreset..")
        already_selected_indices = [self.all_train_im_files.index(i) for i in already_selected]

        if already_selected_indices == []:
            for i in range(self.num_coreset_rounds):
                current_seed = self.seed+i
                coreset_inst, _ = super().get_coreset_inst_and_features_for_round(prev_round_dir, train_logits_path,
                                                                                  seed=current_seed,
                                                                                  delete_preds=delete_preds)
                core_set_indices, _ = coreset_inst.select_batch_(already_selected=already_selected_indices,
                                                                 N=num_samples)
                # get counts for each index
                # save the counts and priors
                # consider if I am going to just sample here and do the ranked selection or something else
                # for now, probably do ranked selection, so it's easy. later probably want to refactor

        # write score file
        with open(im_score_file, "w") as f:
            # higher score for earlier added images
            scores = [score for score in range(len(core_set_indices), 0, -1)]
            for i, im_file in enumerate(self.all_train_im_files):
                if i in core_set_indices:
                    score = scores[core_set_indices.index(i)]
                else:
                    score = 0
                f.write(f"{im_file},{score}\n")

        return [self.all_train_im_files[i] for i in core_set_indices]