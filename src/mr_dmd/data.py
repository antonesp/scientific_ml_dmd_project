from pathlib import Path

import typer
import netCDF4
import numpy as np
from torch.utils.data import Dataset


class ncDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.data = netCDF4.Dataset(self.data_path)

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: str) -> None:
        """Preprocess the raw data and save it to the output folder."""
        sst = self.data["ssf"][:, :, :]
        np.save(output_folder, sst)



def preprocess(data_path: str, output_folder: str) -> None:
    print("Preprocessing data...")
    dataset = ncDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess())
