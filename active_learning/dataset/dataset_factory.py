from active_learning.dataset import ACDC_Dataset, CHAOS_CT_Dataset, LVSC_Dataset, MSCMR_Dataset, DAVIS_Dataset


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_dataset(dataset_type, **dataset_kwargs):
        if dataset_type == "ACDC":
            dataset = ACDC_Dataset(**dataset_kwargs)
        elif dataset_type == "CHAOS_CT":
            dataset = CHAOS_CT_Dataset(**dataset_kwargs)
        elif dataset_type == "LVSC":
            dataset = LVSC_Dataset(**dataset_kwargs)
        elif dataset_type == "MSCMR":
            dataset = MSCMR_Dataset(**dataset_kwargs)
        elif dataset_type == "DAVIS":
            dataset = DAVIS_Dataset(**dataset_kwargs)
        else:
            raise ValueError(f"There is no dataset_type {dataset_type}")
        return dataset
