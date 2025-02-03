# Simple script to train machine learning models on the stress dataset

import os
import argparse

import torch
import numpy as np

from utils.helper_path import CLEANED_DATA_PATH, FEATURE_DATA_PATH, MODELS_PATH, CONFIG_PATH, RESULTS_PATH
from utils.helper_functions import set_seed, get_data_folders, ECGDataset, encode_data, prepare_data, get_ml_model, \
    get_data_balance, evaluate_classifier, create_directory


MODELS_ABBREVIATION_DICT = {
    "lr": "Logistic regression",
    "rf": "Random Forest",
    "dt": "Decision Tree",
    "xgboost": "Extreme Gradient Boosting",
    "lda": "Linear discriminant analysis",
    "qda": "Quadratic discriminant analysis"
}


def validate_scaler(value: str) -> str:
    if value not in ["standard_scaler", "min_max", None]:
        raise argparse.ArgumentTypeError(f"Invalid choice: {value}. Choose from 'standard_scaler' or 'min_max'.")
    return value


def validate_category(value: str) -> str:
    valid_categories = ['high_physical_activity', 'mental_stress', 'baseline',
                        'low_physical_activity', 'moderate_physical_activity']
    if value not in valid_categories:
        raise argparse.ArgumentTypeError(f"Invalid choice: {value}. "
                                         f"Choose from options in {valid_categories}.")
    return value


def validate_ml_model(value: str) -> str:
    valid_ml_models = ['dt', 'rf', 'adaboost', 'lda', 'knn', 'lr', 'xgboost', 'qda']
    if value.lower() not in valid_ml_models:
        raise argparse.ArgumentTypeError(f"Invalid choice: {value}. "
                                         f"Choose from options in {valid_ml_models}.")
    return value


def main(args):
    target_data_path = os.path.join(FEATURE_DATA_PATH, str(args.sample_frequency), str(args.window_size))
    results_path = os.path.join(RESULTS_PATH, str(args.sample_frequency), str(args.window_size), args.model_type.lower())
    create_directory(results_path)

    ecg_dataset = ECGDataset(target_data_path)
    train_data, val_data, test_data = ecg_dataset.get_data()

    train_data, val_data, test_data, feature_names = prepare_data(
        train_data,
        val_data,
        test_data,
        positive_class=args.positive_class,
        negative_class=args.negative_class,
        scaler=args.standard_scaler
    )

    # Get the data balance
    data_balance = get_data_balance(train_data[1], val_data[1], test_data[1])

    # Instantiate the ml model
    ml_model = get_ml_model(args.model_type)

    if args.verbose:
        print(f"Data balance: Class 1: {data_balance}")
        print(f"We fit the model {MODELS_ABBREVIATION_DICT[args.model_type.lower()]}")

    ml_model.fit(train_data[0], train_data[1])

    # Fit the model to the train data and test it on the test data (no hyperparameter tuning for now)
    evaluate_classifier(ml_model, train_data, val_data, test_data, results_path, verbose=args.verbose)

    # Further ToDos:
    #ToDo: Add simple machine learning fit and test
    # Check effect on unseen participants -> training with mixed individuals were we split based on the total data
    # Get hyperparameter tuning pipeline with optuna
    # Get config files, hydra model
    # Can we mash categories together / or get single individual categories?
    # So split up performance by exact categories:
    # Mental stress
    # And the clean split were we split on participants, where generalization matters a lot
    # Get the overall performance (when we aggregate the categories), as well as individual splits
    # Add optuna for hyperparameter tuning
    # wandb logging for tracking experiment?
    # logg results in a results folder
    # saved in a json file?
    # Date
    # Save the best model in a model path
        # Create the model path and directory
    # Fix the windows -> heartbeat should be set dynamically or threshold 40 beats per minute


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed number", default=42, type=int)
    parser.add_argument("--positive_class", help="Which category should be 1", default="mental_stress",
                        type=validate_category)
    parser.add_argument("--negative_class", help="Which category should be 0", default="baseline",
                        type=validate_category)
    parser.add_argument("--standard_scaler", help="Which standard scaler to use. "
                                                  "Choose from 'standard_scaler' or 'min_max'",
                        type=validate_scaler,
                        default="standard_scaler")
    parser.add_argument("--sample_frequency", help="which sample frequency to use for the training",
                        default=1000, type=int)
    parser.add_argument("--window_size", type=int, default=60, help="The window size that we use for detecting stress")
    parser.add_argument("--model_type", help="which model to use"
                                             "Choose from: 'dt', 'rf', 'adaboost', 'lda', "
                                             "'knn', 'lr', 'xgboost', 'qda'",
                        type=validate_ml_model, default="LR")
    parser.add_argument("--verbose", help="Verbose output", action="store_true")
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    main(args)



