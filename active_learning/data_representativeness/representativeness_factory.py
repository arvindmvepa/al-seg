from active_learning.data_representativeness.base_coreset import BaseCoreset, NoCoreset
from active_learning.data_representativeness.kcenter_greedy_coreset import KCenterGreedyCoreset


class RepresentativenessFactory:
    def __init__(self):
        pass 

    @staticmethod
    def create_data_representativeness(representativeness_type, **representativeness_kwargs):
        if representativeness_type == "pass":
            representativeness = NoCoreset(**representativeness_kwargs)
        elif representativeness_type == "kcenter_greedy":
            representativeness = KCenterGreedyCoreset(**representativeness_kwargs)
        else:
            raise ValueError(f"There is no representativeness_type {representativeness_type}")
        return representativeness

