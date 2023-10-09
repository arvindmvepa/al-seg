from abc import ABC, abstractmethod


class BaseDataGeometry(ABC):
    """Abstract class for Coreset sampling"""

    @abstractmethod
    def calculate_representativeness(self, im_score_file, num_samples, round_dir, train_logits_path,
                                     already_selected=[], skip=False,  delete_preds=True, **kwargs):
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

    def calculate_representativeness(self, im_score_file, num_samples, round_dir, train_logits_path,
                                     already_selected=[], skip=False,  delete_preds=True, **kwargs):
        pass