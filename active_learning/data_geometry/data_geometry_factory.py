from active_learning.data_geometry.base_data_geometry import NoDataGeometry
from active_learning.data_geometry.kcenter_greedy_coreset import KCenterGreedyCoreset
from active_learning.data_geometry.coregcn_coreset import CoreGCN
from active_learning.data_geometry.variational_adversarial import VAAL
from active_learning.data_geometry.TypiClust import Typiclust


class DataGeometryFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_data_geometry(data_geometry_type, **geometry_kwargs):
        if data_geometry_type == "pass":
            geometry = NoDataGeometry(**geometry_kwargs)
        elif data_geometry_type == "kcenter_greedy":
            geometry = KCenterGreedyCoreset(**geometry_kwargs)
        elif data_geometry_type == "coregcn":
            geometry = CoreGCN(**geometry_kwargs)
        elif data_geometry_type == "vaal":
            geometry = VAAL(**geometry_kwargs)
        elif data_geometry_type == "typiclust":
            geometry = Typiclust(**geometry_kwargs)
        else:
            raise ValueError(f"There is no geometry_type {data_geometry_type}")
        return geometry
