import numpy as np
import torch

from torch.utils.data import Dataset


class HARDataset(Dataset):
    """
    PyTorch Dataset for HAR sensor windows.

    X shape:
        (samples, channels, timesteps)

    y shape:
        (samples,)
    """

    def __init__(
        self,
        X,
        y,
        label_mapping=None,
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32,
        )

        y = np.asarray(y)

        if label_mapping is None:

            unique_labels = sorted(
                np.unique(y).tolist()
            )

            label_mapping = {
                label: index
                for index, label
                in enumerate(unique_labels)
            }

        self.label_mapping = label_mapping

        encoded_labels = [
            self.label_mapping[label]
            for label in y
        ]

        self.y = torch.tensor(
            encoded_labels,
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        return (
            self.X[index],
            self.y[index],
        )