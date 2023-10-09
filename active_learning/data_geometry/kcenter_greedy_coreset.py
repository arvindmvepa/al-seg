from active_learning.data_geometry.base_coreset import BaseCoreset


class KCenterGreedyCoreset(BaseCoreset):
    """Class for identifying representative data points using Coreset sampling"""

    def __init__(self, patch_size=(256, 256), **kwargs):
        super().__init__(alg_string="kcenter_greedy", patch_size=patch_size, **kwargs)
        
    def calculate_representativeness(self, im_score_file, num_samples, round_dir, train_logits_path,
                                     already_selected=[], skip=False,  delete_preds=True, **kwargs):
        if skip:
            print("Skipping Calculating KCenterGreedyCoreset!")
            return

        print("Calculating KCenterGreedyCoreset..")
        already_selected_indices = [self.all_train_im_files.index(i) for i in already_selected]
        coreset_inst, _ = self.get_coreset_inst_and_features_for_round(round_dir, train_logits_path,
                                                                       delete_preds=delete_preds)
        core_set_indices = coreset_inst.select_batch_(already_selected=already_selected_indices, N=num_samples)

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