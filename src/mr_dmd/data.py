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
        sst = self.data["sst"][:, :, :]
        np.save(output_folder, sst)


def preprocess(data_path: str, output_folder: str) -> None:
    print("Preprocessing data...")
    dataset = ncDataset(data_path)
    dataset.preprocess(output_folder)


def preprocess2(data_path: str, output_folder_data: str, output_folder_mask: str) -> None:
    nc = netCDF4.Dataset(data_path)
    sst = nc["sst"][:, :, :]

    sst_flat = sst.reshape(sst.shape[0], -1).T

    mask = sst_flat.mask[:, 0]

    sst_valid = sst_flat.compressed().reshape(-1, 2115)
    # print(sst_valid.shape)

    # print(type(mask))
    # print(type(sst_valid))
    np.save(output_folder_data, sst_valid)
    np.save(output_folder_mask, mask)


if __name__ == "__main__":
    input_folder = "data/sst.mon.mean.nc"
    output_folder_data = "data/sst"
    output_folder_mask = "data/sst_mask"
    preprocess2(input_folder, output_folder_data, output_folder_mask)
