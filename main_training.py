# Simple script to train machine learning models on the stress dataset

import os
import argparse

import torch
import numpy as np
import optuna
from sklearn import metrics
from optuna.trial import Trial
import json
from datetime import datetime

from utils.helper_path import CLEANED_DATA_PATH, FEATURE_DATA_PATH, MODELS_PATH, CONFIG_PATH, RESULTS_PATH
from utils.helper_functions import set_seed, get_data_folders, ECGDataset, encode_data, prepare_data, get_ml_model, \
    get_data_balance, evaluate_classifier, create_directory, load_yaml_config_file


MODELS_ABBREVIATION_DICT = {
    "lr": "Logistic regression",
    "rf": "Random Forest",
    "dt": "Decision Tree",
    "knn": "K-nearest Neighbor",
    "adaboost": "Adaboost",
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


def objective(trial: Trial, train_data: tuple, val_data: tuple, model_type: str) -> float:
    """
    Objective function for Optuna optimization.
    Returns validation balanced accuracy as the optimization metric.
    """
    # Define hyperparameter search space based on model type
    if model_type.lower() == "lr":
        params = {
            'C': trial.suggest_float('C', 1e-5, 1e2, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 500),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
        }
    elif model_type.lower() == "rf":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
        }
    elif model_type.lower() == "xgboost":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }
    # Add more elif blocks for other models...
    else:
        raise ValueError(f"Hyperparameter optimization not implemented for model type: {model_type}")

    # Create and train model
    model = get_ml_model(model_type, params)
    model.fit(train_data[0], train_data[1])
    
    # Evaluate on validation set
    val_pred = model.predict(val_data[0])
    val_score = metrics.balanced_accuracy_score(val_data[1], val_pred)
    
    return val_score


def main(args):
    target_data_path = os.path.join(FEATURE_DATA_PATH, str(args.sample_frequency), str(args.window_size))
    results_path = os.path.join(RESULTS_PATH, str(args.sample_frequency), str(args.window_size), args.model_type.lower())
    model_config_file = os.path.join(CONFIG_PATH, "models_config_file.yaml")
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

    # Setup for hyperparameter optimization
    study_name = f"{args.model_type.lower()}"
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=args.seed)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, train_data, val_data, args.model_type),
        n_trials=args.n_trials,
        timeout=args.timeout  # in seconds
    )
    
    # Get best parameters and train final model
    best_params = study.best_params
    best_model = get_ml_model(args.model_type, best_params)
    best_model.fit(train_data[0], train_data[1])
    
    # Evaluate final model
    evaluate_classifier(
        best_model, train_data, val_data, test_data,
        save_path=results_path,
        save_name=f"{study_name}_best_performance_results.json",
        verbose=args.verbose)
    
    # Save study statistics and best parameters
    study_stats = {
        "best_params": best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "study_name": study_name,
        "model_type": args.model_type,
        "optimization_history": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ]
    }
    
    with open(os.path.join(results_path, f"{study_name}_optimization_history.json"), "w") as f:
        json.dump(study_stats, f, indent=4)

    if args.verbose:
        print(f"Data balance: Class 1: {data_balance}")
        print(f"We fit the model {MODELS_ABBREVIATION_DICT[args.model_type.lower()]}")

    # Further ToDos:
    #ToDo: Add simple machine learning fit and test
    # Check effect on unseen participants -> training with mixed individuals were we split based on the total data
    # Get hyperparameter tuning pipeline with optuna CHECK
    # Get config files, hydra model
    # Can we mash categories together / or get single individual categories?
    # So split up performance by exact categories:
    # Mental stress
    # And the clean split were we split on participants, where generalization matters a lot
    # Get the overall performance (when we aggregate the categories), as well as individual splits
    # Add optuna for hyperparameter tuning
    # wandb logging for tracking experiment?
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
    parser.add_argument("--n_trials", type=int, default=50,
                       help="Number of optimization trials for Optuna")
    parser.add_argument("--timeout", type=int, default=3600,
                       help="Timeout for optimization in seconds")
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    main(args)



