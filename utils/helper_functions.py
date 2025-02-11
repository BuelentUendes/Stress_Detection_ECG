# Collections of helper functions that are reused across several scripts

import os
import random
import json
import yaml
from typing import Optional, Tuple, Union, Any

import torch
from numpy import ndarray
from optuna import Trial
from pandas import Series, DataFrame
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, clone
from sklearn.utils import resample
from sklearn.dummy import DummyClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from imblearn.over_sampling import ADASYN, SMOTE

from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
import optuna
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


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

        # Here we should split the dataset intro train_feature_selection, val_feature_selection
        train_files_feature_selection, val_files_feature_selection = train_test_split(
            train_files, test_size=0.2
        )

        self.train_feature_selection = self._load_data(train_files_feature_selection)
        self.val_feature_selection = self._load_data(val_files_feature_selection)

        self.train_data = self._load_data(train_files)
        self.val_data = self._load_data(val_files)
        self.test_data = self._load_data(test_files)

    def get_feature_selection_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        returns the datasets split for the feature selection process
        :return:
        """
        return self.train_feature_selection, self.val_feature_selection

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
                 test_data: Optional[pd.DataFrame] = None,
                 positive_class: Optional[str] = "mental_stress",
                 negative_class: Optional[str] = "baseline",
                 resampling_method: Optional[str] = None,
                 scaler: Optional[str] = None,
                 use_subset: Optional[list[bool]] = None) -> tuple:
    """
    Prepares the data for scikit-learn models. Can handle both 2-way (train/val) and 3-way (train/val/test) splits.
    
    Args:
        train_data: DataFrame containing the training data
        val_data: DataFrame containing the validation data
        test_data: Optional DataFrame containing the test data. If None, assumes 2-way split
        positive_class: str, which category to be encoded as 1
        negative_class: str, which category to be encoded as 0
        scaler: StandardScaler instance for normalization
        resampling_method: str, resampling method to use. Options: None, "downsample", "upsample", "smote", "adasyn"
        use_subset: bool, list of bool to indicate which features should be included or not
    
    Returns:
        If test_data is provided:
            Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names)
        If test_data is None:
            Tuple of ((X_train, y_train), (X_val, y_val), feature_names)
    """
    # Handle missing data for provided datasets
    train_data = handle_missing_data(train_data)
    val_data = handle_missing_data(val_data)
    if test_data is not None:
        test_data = handle_missing_data(test_data)

    # Calculate sd1_sd2 feature
    train_data["sd1_sd2"] = train_data["sd1"] / train_data["sd2"]
    val_data["sd1_sd2"] = val_data["sd1"] / val_data["sd2"]
    if test_data is not None:
        test_data["sd1_sd2"] = test_data["sd1"] / test_data["sd2"]

    # Use resampling if provided
    if resampling_method is not None:
        # First encode all available data
        x_train, y_train = encode_data(train_data, positive_class, negative_class)
        x_val, y_val = encode_data(val_data, positive_class, negative_class)

        if test_data is not None:
            x_test, y_test = encode_data(test_data, positive_class, negative_class)

        if use_subset is not None:
            # Ensure the length of use_subset matches the number of features
            assert len(use_subset) == x_train.shape[1], \
                f"Length of use_subset ({len(use_subset)}) must match number of features ({x_train.shape[1]})"
            
            # Filter features using boolean mask
            x_train = x_train.iloc[:, use_subset]
            x_val = x_val.iloc[:, use_subset]
            if test_data is not None:
                x_test = x_test.iloc[:, use_subset]

        # Apply resampling only to training data
        if resampling_method in ["downsample", "upsample"]:
            do_downsampling = resampling_method == "downsample"
            train_data = resample_data(train_data, positive_class, negative_class, downsample=do_downsampling)
            x_train, y_train = encode_data(train_data, positive_class, negative_class)

        elif resampling_method == "smote":
            smote = SMOTE(random_state=42)
            x_train, y_train = smote.fit_resample(x_train, y_train)

        elif resampling_method == "adasyn":
            adasyn = ADASYN(random_state=42)
            x_train, y_train = adasyn.fit_resample(x_train, y_train)
    else:
        # If no resampling, just shuffle and encode the data
        train_data = train_data.sample(frac=1, replace=False, random_state=42).reset_index(drop=True)
        x_train, y_train = encode_data(train_data, positive_class, negative_class)
        x_val, y_val = encode_data(val_data, positive_class, negative_class)
        if test_data is not None:
            x_test, y_test = encode_data(test_data, positive_class, negative_class)

        # Ensure the length of use_subset matches the number of features
        if use_subset is not None:
            assert len(use_subset) == x_train.shape[1], \
                f"Length of use_subset ({len(use_subset)}) must match number of features ({x_train.shape[1]})"

            # Filter features using boolean mask
            x_train = x_train.iloc[:, use_subset]
            x_val = x_val.iloc[:, use_subset]
            if test_data is not None:
                x_test = x_test.iloc[:, use_subset]

    feature_names = list(x_train.columns.values)

    # Apply scaling after resampling if requested
    if scaler is not None:
        assert scaler.lower() in ["min_max", "standard_scaler"], \
            "please set a valid scaler. Options: 'min_max', 'standard_scaler'"
        scaler_obj = StandardScaler() if scaler.lower() == "standard_scaler" else MinMaxScaler()
        x_train = scaler_obj.fit_transform(x_train)
        x_val = scaler_obj.transform(x_val)
        if test_data is not None:
            x_test = scaler_obj.transform(x_test)

    # Return appropriate tuple based on whether test_data was provided
    if test_data is not None:
        return (x_train, y_train), (x_val, y_val), (x_test, y_test), feature_names
    else:
        return (x_train, y_train), (x_val, y_val), feature_names


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
        "svm": {"kernel": "rbf", "C": 1.0, "gamma": 0.7},
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
        "svm": SVC,
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


class FeatureSelectionPipeline:
    """
    Simplified pipeline for feature selection using cross-validation.
    """
    # Set the class attributes
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
        "svm": SVC,
        "random_baseline": DummyClassifier
    }

    def __init__(self, 
                 base_estimator: BaseEstimator,
                 n_features_range: list[int],
                 n_splits: int = 5,
                 n_trials: int = 15,
                 scoring: str = 'roc_auc',
                 random_state: int = 42):
        """
        Initialize the pipeline.
        
        Args:
            base_estimator: Base model for feature selection and final model
            n_features_range: List of number of features to try
            n_splits: Number of cross-validation splits
            n_trials: Number of trials for the bayesian hyperparameter tuning
            scoring: Metric to optimize ('roc_auc' or 'balanced_accuracy')
            random_state: Random seed
        """
        self.base_estimator = base_estimator
        self.n_features_range = n_features_range
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.scoring = scoring
        self.random_state = random_state
        
        self.best_features_mask = None
        self.feature_importance = None
        self.cv_results = None

    def _objective(
            self,
            trial: Trial,
            train_data: tuple,
            val_data: tuple,
            model_type: str,
            metric: str = "roc_auc",
        ) -> float:
        """
        Objective function for Optuna optimization.
        Returns validation balanced accuracy as the optimization metric.
        """
        base_score = np.mean(train_data[1])
        # Define hyperparameter search space based on model type
        if isinstance(self.base_estimator, LogisticRegression):
            params = {
                'C': trial.suggest_float('C', 1e-7, 1e2, log=True),
                'max_iter': 5000,
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'n_jobs': -1,
            }

        elif isinstance(self.base_estimator, RandomForestClassifier):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 5, 25),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 25),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                'n_jobs': -1,
            }
        elif isinstance(self.base_estimator, xgb.XGBClassifier):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 25, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1.0, log=True),
                'base_score': base_score,
                'objective': 'binary:logistic',

                'subsample': trial.suggest_float('subsample', 0.5, 0.8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),

                'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0),

                'use_label_encoder': False,
                'n_jobs': -1
            }
        elif isinstance(self.base_estimator, DecisionTreeClassifier):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
            }
        elif isinstance(self.base_estimator, AdaBoostClassifier):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
                'algorithm': trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R'])
            }
        elif isinstance(self.base_estimator, KNeighborsClassifier):
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 1, 50),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 2),  # 1 for manhattan_distance, 2 for euclidean_distance
                'leaf_size': trial.suggest_int('leaf_size', 20, 50),
                "n_jobs": -1
            }
        elif isinstance(self.base_estimator, LinearDiscriminantAnalysis):
            params = {
                'solver': trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen']),
                'shrinkage': trial.suggest_float('shrinkage', 0.0, 1.0) if trial.suggest_categorical('use_shrinkage',
                                                                                                     [True,
                                                                                                      False]) else None,
                'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True)
            }
        elif isinstance(self.base_estimator, QuadraticDiscriminantAnalysis):
            params = {
                'reg_param': trial.suggest_float('reg_param', 0.0, 1.0),
                'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True)
            }

        elif isinstance(self.base_estimator, SVC):
            params = {
                "C": trial.suggest_float("C", 0.0, 5.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            }

        elif isinstance(self.base_estimator, DummyClassifier):
            params = {
                "strategy": "prior"
            }

        else:
            raise ValueError(f"Hyperparameter optimization not implemented for model type: {model_type}")

        # Create and train model
        model = get_ml_model(model_type, params)
        model.fit(train_data[0], train_data[1])

        # Evaluate on validation set
        if metric == "accuracy":
            val_pred = model.predict(val_data[0])
            val_score = metrics.balanced_accuracy_score(val_data[1], val_pred)
        elif metric == "roc_auc":
            val_score = metrics.roc_auc_score(val_data[1], model.predict_proba(val_data[0])[:, 1])

        return val_score

    def find_best_hyperparameter_base_estimator(self,
                                              train_data: tuple,
                                              val_data: tuple,
                                              n_trials: int = 15,
                                              save_path: Optional[str] = None) -> dict:
        """
        Find best hyperparameters for base estimator using Optuna optimization.
        Results are cached to avoid redundant optimization.
        
        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            n_trials: Number of optimization trials
            save_path: Path to cache hyperparameters. If None, no caching is used.
        
        Returns:
            dict: Best hyperparameters
        """
        # # Check if cached results exist
        # if save_path and os.path.exists(save_path):
        #     with open(save_path, 'r') as f:
        #         return json.load(f)
        
        # Create Optuna study
        study = optuna.create_study(direction="maximize")
        
        # Get model type string from base_estimator class
        model_type = None
        for key, cls in self.model_classes.items():
            if isinstance(self.base_estimator, cls):
                model_type = key
                break
        
        if model_type is None:
            raise ValueError("Unknown base estimator type")
        
        # Define objective function wrapper
        def objective(trial):
            return self._objective(trial, train_data, val_data, model_type, self.scoring)
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        
        # # Cache results if path provided
        # if save_path:
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     with open(save_path, 'w') as f:
        #         json.dump(best_params, f)
        #
        return best_params

    def fit(self, 
            train_data: tuple[np.ndarray, np.ndarray],
            val_data: tuple[np.ndarray, np.ndarray],
            feature_names: list[str] = None,
            save_path: Optional[str] = None) -> None:
        """
        Fit the feature selection pipeline using provided train/val sets.
        
        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            feature_names: List of feature names
            save_path: Path to cache hyperparameters. If None, no saving is done.
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # First find best hyperparameters
        best_params = self.find_best_hyperparameter_base_estimator(
            train_data,
            val_data,
            n_trials=self.n_trials,
            save_path=save_path
        )
        
        # Create optimized base estimator
        optimized_estimator = clone(self.base_estimator).set_params(**best_params)
        
        # Try each number of features
        scores = []
        feature_importance_scores = np.zeros(X_train.shape[1])
        selected_features_count = np.zeros(X_train.shape[1])
        
        for n_features in self.n_features_range:
            print(f"Evaluating {n_features} features")
            
            # Feature selection using optimized estimator
            rfe = RFE(
                estimator=clone(optimized_estimator),
                n_features_to_select=n_features
            )
            
            # Create and fit pipeline
            pipeline = Pipeline([
                ('rfe', rfe),
                ('model', clone(optimized_estimator))
            ])
            print(f"Fitting estimator")
            pipeline.fit(X_train, y_train)
            
            # Score on validation set
            if self.scoring == 'roc_auc':
                score = roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])
            else:
                score = balanced_accuracy_score(y_val, pipeline.predict(X_val))
            print(f"The score is {score}")
            scores.append(score)
            
            # Track feature importance
            feature_importance_scores += rfe.ranking_
            selected_features_count += rfe.support_
        
        # Find best number of features
        best_n_features = self.n_features_range[np.argmax(scores)]
        
        # Final feature selection with best number of features
        final_rfe = RFE(
            estimator=clone(optimized_estimator),
            n_features_to_select=best_n_features
        )
        final_rfe.fit(X_train, y_train)
        
        # Store results
        self.best_features_mask = final_rfe.support_
        self.cv_results = {
            'scores': [float(score) for score in scores],  # Convert numpy floats to Python floats
            'best_score': float(np.max(scores)),  # Convert numpy float to Python float
            'best_n_features': int(best_n_features),  # Convert numpy int to Python int
            'best_params': best_params,
            'feature_selection_mask': [bool(mask) for mask in self.best_features_mask],  # Convert numpy bools to Python bools
        }
        self.feature_importance = {
            str(name): {  # Ensure keys are strings
                'importance_score': float(score),  # Convert numpy float to Python float
                'selected': bool(count)  # Convert numpy bool to Python bool
            }
            for name, score, count in zip(
                feature_names,
                feature_importance_scores,
                selected_features_count
            )
        }

        with open(os.path.join(save_path, "feature_selection_results.json"), 'w') as f:
            json.dump(self.cv_results, f, indent=4)  # Save results in JSON format

        with open(os.path.join(save_path, "feature_importance_report.json"), "w") as f:
            json.dump(self.feature_importance, f, indent=4)


        
