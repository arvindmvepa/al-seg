from active_learning.data_geometry.base_data_geometry import NoDataGeometry
from active_learning.data_geometry.kcenter_greedy_coreset import KCenterGreedyCoreset


class DataGeometryFactory:
    def __init__(self):
        pass 

    @staticmethod
    def create_data_geometry(data_geometry_type, **geometry_kwargs):
        if data_geometry_type == "pass":
            geometry = NoDataGeometry(**geometry_kwargs)
        elif data_geometry_type == "kcenter_greedy":
            geometry = KCenterGreedyCoreset(**geometry_kwargs)
        else:
            raise ValueError(f"There is no geometry_type {data_geometry_type}")
        return geometry

