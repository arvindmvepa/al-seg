from abc import ABC, abstractmethod


class BaseModelUncertainty(ABC):
    """Abstract class for ModelUncertainty"""

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def calculate_uncertainty(self, im_score_file, round_dir, ignore_ims_dict, skip=False, **kwargs):
        raise NotImplementedError()


class NoModelUncertainty(BaseModelUncertainty):
    """Placeholder class for no model uncertainty"""

    def calculate_uncertainty(self, im_score_file, round_dir, ignore_ims_dict, skip=False, **kwargs):
        pass