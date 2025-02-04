# Collections of helper functions that are reused across several scripts

import os
import random
import json
import yaml
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.utils import resample

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


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

    def __init__(self, root_dir: str, test_size: Optional[float] = 0.2, val_size: Optional[float] = 0.2):
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


#Todo: Extend to multiclass classification
def encode_data(data: pd.DataFrame, positive_class: str, negative_class: str) -> pd.DataFrame:
    # First drop data that is not either in the positive class or negative class
    data = data[(data['category'] == positive_class) | (data['category'] == negative_class)]  # Filter relevant classes
    # Then label the data 1 for positive and 0 for negative
    data.loc[:, 'category'] = data['category'].apply(lambda x: 1 if x == positive_class else 0)  # Encode classes

    # Split data into x_data and y_data
    x = data.drop(columns=["category"])
    # The target label needs to be an integer
    y = data["category"].astype(int)

    return x,y


def handle_missing_data(data: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with infinite values and NaN values
    original_data_len = len(data)
    data = data[~data.isin([np.inf, -np.inf]).any(axis=1)]  # Drop rows with infinite values
    data = data.dropna()  # More efficient way to drop rows with NaN values

    # Assert statements to check if it worked
    assert not data.isin([np.inf, -np.inf]).any().any(), "infinity data is still detected"  # Check for infinite values
    assert not data.isna().any().any(), "np.nan data is still detected"  # Check for NaN values

    dropped_percent = ((original_data_len - len(data)) / original_data_len) * 100
    print(f"We dropped {np.round(dropped_percent, 4)} percent of the original data")

    return data


def downsample_majority_class(data: pd.DataFrame, positive_class: str, negative_class: str) -> pd.DataFrame:
    """
    Downsamples the majority class to match the size of the minority class.
    
    Args:
        data (pd.DataFrame): Input DataFrame containing the data
        positive_class (str): Label of the positive class
        negative_class (str): Label of the negative class
    
    Returns:
        pd.DataFrame: Balanced DataFrame with downsampled majority class
    """
    df_positive_class = data[data["category"] == positive_class]
    df_negative_class = data[data["category"] == negative_class]

    # We need to downsample the majority class
    if len(df_positive_class) >= len(df_negative_class):
        positive_class_downsampled = resample(df_positive_class, 
                                            replace=False, 
                                            n_samples=len(df_negative_class),
                                            random_state=42)  
        balanced_data = pd.concat([positive_class_downsampled, df_negative_class])
    else:
        negative_class_downsampled = resample(df_negative_class, 
                                            replace=False, 
                                            n_samples=len(df_positive_class),
                                            random_state=42)  
        balanced_data = pd.concat([negative_class_downsampled, df_positive_class])

    # Verify the balancing worked
    assert len(balanced_data[balanced_data["category"] == positive_class]) == \
           len(balanced_data[balanced_data["category"] == negative_class]), \
           "Downsampling failed: classes are not balanced"

    return balanced_data


def prepare_data(train_data: pd.DataFrame,
                 val_data: pd.DataFrame,
                 test_data: pd.DataFrame,
                 positive_class: Optional[str] = "mental_stress",
                 negative_class: Optional[str] = "baseline",
                 use_downsampling: Optional[bool] = False,
                 scaler: StandardScaler = None) -> tuple:
    """
    Prepares the data for scikit-learn models.
    :param train_data: DataFrame containing the training data
    :param val_data: DataFrame containing the validation data
    :param test_data: DataFrame containing the test data
    :param positive_class str: which category to be encoded as 1
    :param negative_class str: which category to be encoded as 0
    :param scaler: StandardScaler instance for normalization
    :param use_downsampling: bool, if set, we downsample the majority class. Default False
    :return: Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """

    # We first handle missing data
    train_data = handle_missing_data(train_data)
    val_data = handle_missing_data(val_data)
    test_data = handle_missing_data(test_data)

    # Downsample the majority class if necessary
    if use_downsampling:
        train_data = downsample_majority_class(train_data, positive_class, negative_class)

    # Get the columns
    x_train, y_train = encode_data(train_data, positive_class, negative_class)
    x_val, y_val = encode_data(val_data, positive_class, negative_class)
    x_test, y_test = encode_data(test_data, positive_class, negative_class)

    feature_names = list(x_train.columns.values)

    if scaler is not None:
        assert scaler.lower() in ["min_max", "standard_scaler"], \
            "please set a valid scaler. Options: 'min_max', 'standard_scaler'"
        scaler = StandardScaler() if scaler.lower() == "standard_scaler" else MinMaxScaler()
        x_train = scaler.fit_transform(x_train)  # Fit and transform on training data
        x_val = scaler.transform(x_val)  # Transform validation data
        x_test = scaler.transform(x_test)  # Transform test data

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), feature_names


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


def get_ml_model(model: str, params: dict = None):
    """
    Returns the machine learning model initialized with the specified configuration settings.

    Args:
        model (str): The name of the machine learning model to initialize. 
                     Options include 'DT', 'RF', 'AdaBoost', 'LDA', 'KNN', 'LR', 'XGBoost', 'QDA'.
        params (dict, optional): A dictionary of parameters to initialize the model. 
                                 If None, default parameters will be used.

    Raises:
        ValueError: If the specified model name is invalid.

    Returns:
        object: An instance of the specified machine learning model initialized with the given parameters.
    """
    # Default parameters for each model
    default_params = {
        "dt": {"random_state": 42},
        "rf": {"random_state": 42, "bootstrap": False, "n_jobs": -1},
        "adaboost": {"base_estimator": DecisionTreeClassifier(criterion='entropy', min_samples_split=20)},
        "lda": {},
        "knn": {"n_jobs": -1},
        "lr": {"n_jobs": -1},
        "xgboost": {},
        "qda": {}
    }

    # Map model names to their corresponding classes
    model_classes = {
        "dt": DecisionTreeClassifier,
        "rf": RandomForestClassifier,
        "adaboost": AdaBoostClassifier,
        "lda": LinearDiscriminantAnalysis,
        "knn": KNeighborsClassifier,
        "lr": LogisticRegression,
        "xgboost": GradientBoostingClassifier,
        "qda": QuadraticDiscriminantAnalysis
    }

    if model.lower() not in model_classes:
        raise ValueError('Invalid model')

    if params is None:
        params = default_params[model.lower()]

    cls = model_classes[model.lower()](**params)  # Initialize the model with parameters

    return cls


def get_data_balance(train_data:np.array, val_data: np.array, test_data: np.array) -> np.array:
    """
    Calculates the imbalance of the dataset overall
    """

    overall_data_len = len(train_data) + len(val_data) + len(test_data)
    percentage_train = len(train_data) / overall_data_len
    percentage_val = len(val_data) / overall_data_len
    percentage_test = len(test_data) / overall_data_len

    assert_almost_equal((percentage_train + percentage_val + percentage_test), 1.0, decimal=5)

    class_1_train = np.mean(train_data)
    class_1_val = np.mean(val_data)
    class_1_test = np.mean(test_data)

    data_balance = np.round(percentage_train * class_1_train + percentage_val * class_1_val +  percentage_test * class_1_test, 4)
    return data_balance


def evaluate_classifier(ml_model: BaseEstimator,
                        train_data: tuple[np.ndarray, np.ndarray],
                        val_data: tuple[np.ndarray, np.ndarray],
                        test_data: tuple[np.ndarray, np.ndarray],
                        save_path: str,
                        save_name: str,
                        verbose: bool = False) -> dict[str, float]:
    """
    Evaluates the trained machine learning model and gets the performance metrics
    :param ml_model: scikit-learn model
    :param train_data: tuple, with 0 being the x_data and 1 the labels
    :param val_data: tuple, with 0 being the x_data and 1 the labels
    :param test_data: tuple, with 0 being the x_data and 1 the labels
    :param save_path: str, path where to save the results
    :param verbose: bool, flag for verbose output
    :param save_name: str, name of the json file
    :return: dictionary with the performance metrics
    """

    def round_result(value: float) -> float:
        return np.round(value, 4)

    results = {
        'proportion class 1': get_data_balance(train_data[1], val_data[1], test_data[1]),
        'train_accuracy': round_result(metrics.accuracy_score(train_data[1], ml_model.predict(train_data[0]))),
        'train_balanced_accuracy': round_result(metrics.balanced_accuracy_score(train_data[1], ml_model.predict(train_data[0]))),
        'val_accuracy': round_result(metrics.accuracy_score(val_data[1], ml_model.predict(val_data[0]))),
        'val_balanced_accuracy': round_result(metrics.balanced_accuracy_score(val_data[1], ml_model.predict(val_data[0]))),
        'test_accuracy': round_result(metrics.accuracy_score(test_data[1], ml_model.predict(test_data[0]))),
        'test_balanced_accuracy': round_result(metrics.balanced_accuracy_score(test_data[1], ml_model.predict(test_data[0]))),
    }

    # Binary classification
    if len(train_data[1].unique()) == 2:
        results['train_f1'] = round_result(metrics.f1_score(train_data[1], ml_model.predict(train_data[0])))
        results['val_f1'] = round_result(metrics.f1_score(val_data[1], ml_model.predict(val_data[0])))
        results['test_f1'] = round_result(metrics.f1_score(test_data[1], ml_model.predict(test_data[0])))

        # ROC AUC
        results['train_roc_auc'] = round_result(metrics.roc_auc_score(train_data[1], ml_model.predict_proba(train_data[0])[:, 1]))
        results['val_roc_auc'] = round_result(metrics.roc_auc_score(val_data[1], ml_model.predict_proba(val_data[0])[:, 1]))
        results['test_roc_auc'] = round_result(metrics.roc_auc_score(test_data[1], ml_model.predict_proba(test_data[0])[:, 1]))

    else:
        raise NotImplementedError("We have not yet implemented multiclass classification")

    if verbose:
        print(results)

    # Save results to a JSON file
    if save_name is None:
        save_name = "performance_metrics.json"

    with open(os.path.join(save_path, save_name), 'w') as f:
        json.dump(results, f)  # Save results in JSON format

    return results


def load_yaml_config_file(path_to_yaml_file: str):
    """
    Loads a yaml file
    :param path_to_yaml_file:
    :return: the resulting dictionary
    """
    try:
        with open(path_to_yaml_file) as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print("We could not find the yaml file that you specified")



        
