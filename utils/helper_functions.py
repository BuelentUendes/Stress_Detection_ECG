# Collections of helper functions that are reused across several scripts

import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


def create_directory(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        path: Path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def set_seed(seed_number: int) -> None:
    """
    Set the seed
    :param seed_number: seed number to set
    :return: None
    """
    # Set seed for reproducibility
    torch.manual_seed(seed_number)
    np.random.seed(seed_number)
    random.seed(seed_number)


def get_data_folders(input_path: str) -> list[str]:
    """
    Loads the data and from the input_path
    :param input_path: input_path to load the data from
    :return: list of participants
    """
    data_folders = [filename for filename in os.listdir(input_path) if filename.lower().endswith((".csv"))]
    return data_folders


class ECGDataset(Dataset):
    """
    Feature engineered dataset for the ECG dataset
    """

    def __init__(self, root_dir: str, transform=None):
        """
        :param csv_files: list of strings
        :param root_dir: root directory for the data import
        :param transform: which transformation to apply to the dataset
        """

        self.root_dir = root_dir
        self.transform = transform
        self._get_data_folders()
        self.data = self._load_data()

    def _get_data_folders(self) -> None:
        """
        Gets the data folders of the root directory which we will then load as a dataset
        """
        self.data_folders = [filename for filename in os.listdir(self.root_dir) if filename.lower().endswith((".csv"))]

    def _load_data(self) -> pd.DataFrame:
        """
        Loads the data into a pandas dataframe from the CSV files.
        :return: Combined DataFrame containing data from all CSV files.
        """
        dataframes = []  # List to hold individual DataFrames

        for csv_file in self.data_folders:
            file_path = os.path.join(self.root_dir, csv_file)  # Construct full file path
            try:
                df = pd.read_csv(file_path)  # Read CSV file into DataFrame
                dataframes.append(df)  # Append DataFrame to the list
            except Exception as e:
                print(f"Error reading {file_path}: {e}")  # Handle exceptions

        combined_df = pd.concat(dataframes, ignore_index=True)  # Concatenate all DataFrames
        return combined_df  # Return the combined DataFrame

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        :return: Number of samples
        """
        self.data = self._load_data()  # Load data if not already loaded
        return len(self.data)  # Return the number of rows in the DataFrame

    def __getitem__(self, index: int) -> pd.Series:
        """
        Retrieves a sample from the dataset.
        :param index: Index of the sample to retrieve
        :return: A single sample as a pandas Series
        """
        if not hasattr(self, 'data'):
            self.data = self._load_data()  # Load data if not already loaded

        return self.data.iloc[index]  # Return the sample at the specified index

        
