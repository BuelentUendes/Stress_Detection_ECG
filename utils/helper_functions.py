# Collections of helper functions that are reused across several scripts

import os
import random
import json
import yaml
from typing import Optional, Tuple, Union, Any

import torch
from numpy import ndarray
from pandas import Series, DataFrame
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.utils import resample
from sklearn.dummy import DummyClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from imblearn.over_sampling import ADASYN, SMOTE


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
def encode_data(data: pd.DataFrame, positive_class: str, negative_class: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # First drop data that is not either in the positive class or negative class
    data = data[(data['category'] == positive_class) | (data['category'] == negative_class)]  # Filter relevant classes
    # Then label the data 1 for positive and 0 for negative
    data.loc[:, 'category'] = data['category'].apply(lambda x: 1 if x == positive_class else 0)  # Encode classes

    # Split data into x_data and y_data
    x = data.drop(columns=["category"]).reset_index(drop=True)
    # The target label needs to be an integer
    y = data["category"].astype(int).reset_index(drop=True)

    return x, y


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


def resample_data(data: pd.DataFrame,
                  positive_class: str,
                  negative_class: str,
                  downsample: bool) -> pd.DataFrame:
    """
    Resample the data to balance classes, either via upsampling minority or downsampling majority.
    
    Args:
        data: Input DataFrame containing the data
        positive_class: Label of the positive class
        negative_class: Label of the negative class
        downsample: If True, downsample majority class; if False, upsample minority

    Returns:
        Balanced DataFrame with equal class distributions

    Note:
        Upsampling minority class will create duplicates for highly imbalanced data
    """
    # Split data by class
    df_positive = data[data["category"] == positive_class]
    df_negative = data[data["category"] == negative_class]
    
    # Determine majority and minority classes
    if len(df_positive) >= len(df_negative):
        majority_df, minority_df = df_positive, df_negative
    else:
        majority_df, minority_df = df_negative, df_positive

    # Perform resampling
    if downsample:
        print(f"We downsample!")
        resampled_majority = resample(majority_df,
                                    replace=False,
                                    n_samples=len(minority_df),
                                    random_state=42)
        balanced_data = pd.concat([resampled_majority, minority_df])
    else:
        print(f"We upsample!")
        resampled_minority = resample(minority_df,
                                    replace=True,
                                    n_samples=len(majority_df),
                                    random_state=42)
        balanced_data = pd.concat([resampled_minority, majority_df])

    # Shuffle the final dataset
    balanced_data = balanced_data.sample(frac=1, replace=False, random_state=42).reset_index(drop=True)

    # Verify balancing
    class_counts = balanced_data["category"].value_counts()
    assert class_counts[positive_class] == class_counts[negative_class], \
        f"Resampling failed: classes are not balanced. Counts: {class_counts}"

    return balanced_data


def prepare_data(train_data: pd.DataFrame,
                 val_data: pd.DataFrame,
                 test_data: pd.DataFrame,
                 positive_class: Optional[str] = "mental_stress",
                 negative_class: Optional[str] = "baseline",
                 resampling_method: Optional[str] = None,
                 scaler: StandardScaler = None) -> tuple:
    """
    Prepares the data for scikit-learn models.
    :param train_data: DataFrame containing the training data
    :param val_data: DataFrame containing the validation data
    :param test_data: DataFrame containing the test data
    :param positive_class: str, which category to be encoded as 1
    :param negative_class: str, which category to be encoded as 0
    :param scaler: StandardScaler instance for normalization
    :param resampling_method: bool, if set, we downsample the majority class. Default False
    :return: Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """

    # We first handle missing data
    train_data = handle_missing_data(train_data)
    val_data = handle_missing_data(val_data)
    test_data = handle_missing_data(test_data)

    # Use resampling if provided
    if resampling_method is not None:
        # First encode the data before resampling
        x_train, y_train = encode_data(train_data, positive_class, negative_class)
        x_val, y_val = encode_data(val_data, positive_class, negative_class)
        x_test, y_test = encode_data(test_data, positive_class, negative_class)
        
        if resampling_method in ["downsample", "upsample"]:
            do_downsampling = True if resampling_method == "downsample" else False
            train_data = resample_data(train_data, positive_class, negative_class, downsample=do_downsampling)
            # Re-encode the resampled data
            x_train, y_train = encode_data(train_data, positive_class, negative_class)
        elif resampling_method == "smote":
            smote = SMOTE(random_state=42, n_jobs=-1)
            x_train, y_train = smote.fit_resample(x_train, y_train)
        elif resampling_method == "adasyn":
            adasyn = ADASYN(random_state=42, n_jobs=-1)
            x_train, y_train = adasyn.fit_resample(x_train, y_train)
    else:
        # If no resampling, just encode the data normally
        # Shuffle the data
        # Shuffle the final training data
        train_data = train_data.sample(frac=1, replace=False, random_state=42).reset_index(drop=True)

        x_train, y_train = encode_data(train_data, positive_class, negative_class)
        x_val, y_val = encode_data(val_data, positive_class, negative_class)
        x_test, y_test = encode_data(test_data, positive_class, negative_class)

    feature_names = list(x_train.columns.values)

    # Apply scaling after resampling if requested
    # I guess here the idx issue happens
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
        "qda": {},
        "random_baseline": {"strategy": "prior"},
    }

    # Map model names to their corresponding classes
    model_classes = {
        "dt": DecisionTreeClassifier,
        "rf": RandomForestClassifier,
        "adaboost": AdaBoostClassifier,
        "lda": LinearDiscriminantAnalysis,
        "knn": KNeighborsClassifier,
        "lr": LogisticRegression,
        # "xgboost": GradientBoostingClassifier,
        "xgboost": xgb.XGBClassifier,
        "qda": QuadraticDiscriminantAnalysis,
        "random_baseline": DummyClassifier
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

    def get_pr_curve(y_true:np.array, y_score: np.array) -> float:
        pr_auc = metrics.average_precision_score(y_true, y_score)
        return pr_auc
    

    results = {
        'proportion class 1': get_data_balance(train_data[1], val_data[1], test_data[1]),
        'train_balanced_accuracy': round_result(metrics.balanced_accuracy_score(train_data[1], ml_model.predict(train_data[0]))),
        'val_balanced_accuracy': round_result(metrics.balanced_accuracy_score(val_data[1], ml_model.predict(val_data[0]))),
        'test_balanced_accuracy': round_result(metrics.balanced_accuracy_score(test_data[1], ml_model.predict(test_data[0]))),
    }

    # Binary classification
    if len(train_data[1].unique()) == 2:
        # Add here the PR-recall curve! instead of F1
        results['train_pr_auc'] = round_result(get_pr_curve(train_data[1], ml_model.predict_proba(train_data[0])[:, 1]))
        results['val_pr_auc'] = round_result(get_pr_curve(val_data[1], ml_model.predict_proba(val_data[0])[:, 1]))
        results['test_pr_auc'] = round_result(get_pr_curve(test_data[1], ml_model.predict_proba(test_data[0])[:, 1]))

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


def bootstrap_test_performance(model: BaseEstimator,
                               test_data: tuple[np.ndarray, np.ndarray],
                               bootstrap_samples: int,
                               bootstrap_method: str) -> dict[str, float]:
    """
    Performs bootstrap resampling to estimate model performance metrics and their confidence intervals.
    
    This function repeatedly samples the test data with replacement to create bootstrap samples,
    evaluates the model on each sample, and calculates performance metrics along with their
    95% confidence intervals.
    
    Args:
        model: Trained classifier model that implements predict_proba and predict methods
        test_data: tuple of (X_test, y_test) containing:
            - X_test: array-like of shape (n_samples, n_features)
            - y_test: array-like of shape (n_samples,) with true labels
        bootstrap_samples: int, number of bootstrap iterations (default: 1000)
        bootstrap_method: str, which method to use for bootstrap samples.
    
    Returns:
        dict: Dictionary containing performance metrics and their confidence intervals:
            {
                'roc_auc': {'mean': float, 'ci_lower': float, 'ci_upper': float},
                'pr_auc': {'mean': float, 'ci_lower': float, 'ci_upper': float},
                'balanced_accuracy': {'mean': float, 'ci_lower': float, 'ci_upper': float}
            }
    """
    X_test, y_test = test_data
    n_samples = len(X_test)
    
    # Initialize results dictionary
    results = {
        'roc_auc': [],
        'pr_auc': [],
        'balanced_accuracy': []
    }
    
    for _ in range(bootstrap_samples):
        # Resample the dataset with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X_test[indices]
        y_bootstrap = y_test[indices]
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_bootstrap)[:, 1]
        y_pred = model.predict(X_bootstrap)
        
        # Calculate metrics
        results['roc_auc'].append(metrics.roc_auc_score(y_bootstrap, y_pred_proba))
        results['pr_auc'].append(metrics.average_precision_score(y_bootstrap, y_pred_proba))
        results['balanced_accuracy'].append(metrics.balanced_accuracy_score(y_bootstrap, y_pred))
    
    # Calculate confidence intervals and means
    final_results = {}
    for metric in results.keys():
        values = np.array(results[metric])
        mean_val = np.mean(values)
        if bootstrap_method == "quantile":
            ci_lower = np.percentile(values, 2.5)  # 2.5th percentile for lower bound
            ci_upper = np.percentile(values, 97.5)  # 97.5th percentile for upper bound
        else:
            raise NotImplementedError("We have not implemented 'se' and 'BCa'")

        final_results[metric] = {
            'mean': np.round(mean_val, 4),
            'ci_lower': np.round(ci_lower, 4),
            'ci_upper': np.round(ci_upper, 4)
        }
    
    return final_results


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



        
