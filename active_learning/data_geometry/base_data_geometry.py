from abc import ABC, abstractmethod


class BaseDataGeometry(ABC):
    """Abstract class for sampling using Data Geometry"""

    @abstractmethod
    def calculate_representativeness(self, im_score_file, num_samples, already_selected=[], skip=False, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def setup(self, data_root, all_train_im_files):
        raise NotImplementedError()

    

class NoDataGeometry(BaseDataGeometry):
    """Placeholder class for no data geometry"""
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def setup(self, data_root, all_train_im_files):
        pass

    def calculate_representativeness(self, im_score_file, num_samples, already_selected=[], skip=False, **kwargs):
        pass