from torch.utils.data import Dataset


class DatasetWrapper(Dataset):

    def __init__(self, image_features, transform=None):
        self.image_features = image_features
        self.transform = transform

    def __len__(self):
        return len(self.image_features)

    def __getitem__(self, idx):
        sample = self.image_features[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample