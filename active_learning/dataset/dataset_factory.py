from active_learning.dataset.acdc_dataset import ACDCDataset


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_dataset(dataset_type, **dataset_kwargs):
        if dataset_type == "ACDC":
            dataset = ACDCDataset(**dataset_kwargs)
        else:
            raise ValueError(f"There is no dataset_type {dataset_type}")
        return dataset
