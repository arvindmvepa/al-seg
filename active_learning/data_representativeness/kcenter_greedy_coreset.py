import numpy as np
from active_learning.data_representativeness.base_coreset import BaseCoreset
from active_learning.data_representativeness.kcenter_greedy import kCenterGreedy


class KCenterGreedyCoreset(BaseCoreset):
    """Class for identifying representative data points using Coreset sampling
    """

    def __init__(self, im_path_list=None, patch_size=(256, 256), **kwargs):
        super().__init__(im_path_list=im_path_list, patch_size=patch_size)
        self.pass_ = False
        
    def calculate_representativeness(self, num_samples, already_selected=[], skip=False, **kwargs):
        if skip:
            print("Skipping Calculating Representativeness!")
            return
        if self.X is None:
            raise ValueError("X is empty! Please initialize X before calculating representativeness.")
        KCG = kCenterGreedy(self.X)
        core_set = KCG.select_batch_(already_selected=already_selected, N=num_samples)
        return core_set