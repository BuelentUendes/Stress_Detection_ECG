# Collections of helper functions that are reused across several scripts

import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


class ECGDataset:
    """
    Feature engineered dataset for the ECG dataset
    """

    def __init__(self, root_dir: str, test_size: float = 0.2, val_size: float = 0.2):
        """
        :param root_dir: root directory for the data import
        :param test_size: test size split, default 0.2
        :param val_size: val size split, default 0.2
        """

        assert isinstance(test_size, float), "test size needs to be a float"
        assert 0.0 <= test_size <= 1.0, "test size should be in between 0 and 1"

        assert isinstance(val_size, float), "test size needs to be a float"
        assert 0.0 <= val_size <= 1.0, "val size should be in between 0 and 1"

        self.root_dir = root_dir
        self.test_size = test_size
        self.val_size = val_size

        self._get_data_folders()
        self._split_data()

    def _get_data_folders(self) -> None:
        """
        Gets the data folders of the root directory which we will then load as a dataset
        """
        self.data_folders = [filename for filename in os.listdir(self.root_dir) if filename.lower().endswith((".csv"))]

    def _load_data(self, data_files: list[str]) -> pd.DataFrame:
        """
        Loads the data into a pandas dataframe from the CSV files.
        :param data_files: list of data files to load from
        :return: Combined DataFrame containing data from all CSV files.
        """
        dataframes = []  # List to hold individual DataFrames

        for csv_file in data_files:
            file_path = os.path.join(self.root_dir, csv_file)  # Construct full file path
            try:
                df = pd.read_csv(file_path)  # Read CSV file into DataFrame
                dataframes.append(df)  # Append DataFrame to the list
            except Exception as e:
                print(f"Error reading {file_path}: {e}")  # Handle exceptions

        combined_df = pd.concat(dataframes, ignore_index=True)  # Concatenate all DataFrames
        return combined_df  # Return the combined DataFrame

    # def __len__(self) -> int:
    #     """
    #     Returns the total number of samples in the dataset.
    #     :return: Number of samples
    #     """
    #     self.data = self._load_data()  # Load data if not already loaded
    #     return len(self.data)  # Return the number of rows in the DataFrame
    #
    # def __getitem__(self, index: int) -> pd.Series:
    #     """
    #     Retrieves a sample from the dataset.
    #     :param index: Index of the sample to retrieve
    #     :return: A single sample as a pandas Series
    #     """
    #     if not hasattr(self, 'data'):
    #         self.data = self._load_data()  # Load data if not already loaded
    #
    #     return self.data.iloc[index]  # Return the sample at the specified index

    def _split_data(self) -> tuple:
        """
        Splits the dataset into train, validation, and test sets based on participant CSV files.
        :param test_size: Proportion of the dataset to include in the test split
        :param val_size: Proportion of the dataset to include in the validation split
        :return: Tuple of (train_data, val_data, test_data)
        """
        # Use the filenames as participant identifiers
        participant_files = self.data_folders
        train_files, test_files = train_test_split(participant_files, test_size=self.test_size)
        val_size_adjusted = self.val_size / (1 - self.test_size)  # Adjust validation size based on remaining data
        train_files, val_files = train_test_split(train_files, test_size=val_size_adjusted)

        self.train_data = self._load_data(train_files)
        self.val_data = self._load_data(val_files)
        self.test_data = self._load_data(test_files)

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        returns the datasets split in train, val and test data
        :return:
        """
        return self.train_data, self.val_data, self.test_data


def encode_data(data: pd.DataFrame, positive_class: str, negative_class: str) -> pd.DataFrame:

    category_list = list(set(data["category"].values))
    assert positive_class in category_list, f"positive class needs to be one in {category_list}"
    assert negative_class in category_list, f"negative class needs to be one in {category_list}"

    # First drop data that is not either in the positive class or negative class
    data = data[(data['category'] == positive_class) | (data['category'] == negative_class)]  # Filter relevant classes
    # Then label the data 1 for positive and 0 for negative
    data['category'] = data['category'].apply(lambda x: 1 if x == positive_class else 0)  # Encode classes
    return data


# Function to prepare data for scikit-learn
def prepare_data(train_data: pd.DataFrame,
                 val_data: pd.DataFrame,
                 test_data: pd.DataFrame,
                 positive_class: str = "mental_stress",
                 negative_class: str = "baseline",
                 scaler: StandardScaler = None) -> tuple:
    """
    Prepares the data for scikit-learn models.
    :param train_data: DataFrame containing the training data
    :param val_data: DataFrame containing the validation data
    :param test_data: DataFrame containing the test data
    :param positive_class str: which category to be encoded as 1
    :param negative_class str: which category to be encoded as 0
    :param scaler: StandardScaler instance for normalization
    :return: Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """






def normalize_data(train_data: pd.DataFrame) -> tuple:
    """
    Normalizes the training data and returns the scaler for future use.
    :param train_data: DataFrame containing the training data
    :return: Tuple of (normalized_train_data, scaler)
    """
    scaler = StandardScaler()
    features = train_data.drop(columns=['target'])  # Replace 'target' with your actual target column name
    normalized_train_data = scaler.fit_transform(features)  # Normalize features
    return normalized_train_data, scaler


# Function to prepare data for scikit-learn
def prepare_data(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame, scaler: StandardScaler) -> tuple:
    """
    Prepares the data for scikit-learn models.
    :param train_data: DataFrame containing the training data
    :param val_data: DataFrame containing the validation data
    :param test_data: DataFrame containing the test data
    :param scaler: StandardScaler instance for normalization
    :return: Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    X_train = scaler.transform(train_data.drop(columns=['target']))  # Replace 'target' with your actual target column name
    y_train = train_data['target']  # Replace 'target' with your actual target column name
    X_val = scaler.transform(val_data.drop(columns=['target']))  # Replace 'target' with your actual target column name
    y_val = val_data['target']  # Replace 'target' with your actual target column name
    X_test = scaler.transform(test_data.drop(columns=['target']))  # Replace 'target' with your actual target column name
    y_test = test_data['target']  # Replace 'target' with your actual target column name

    return X_train, y_train, X_val, y_val, X_test, y_test






# Training pipeline for scikit-learn models
def train_sklearn_model(model, X_train, y_train, X_val, y_val):
    """
    Trains a scikit-learn model using the provided training and validation data.
    :param model: The scikit-learn model to train
    :param X_train: Features for the training set
    :param y_train: Target for the training set
    :param X_val: Features for the validation set
    :param y_val: Target for the validation set
    """
    model.fit(X_train, y_train)  # Fit the model on the training data
    val_score = model.score(X_val, y_val)  # Evaluate on the validation set
    print(f'Validation Score: {val_score}')

        
