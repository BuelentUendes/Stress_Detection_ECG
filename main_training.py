# Simple script to train machine learning models on the stress dataset

import os
import argparse
from typing import Any
import warnings
warnings.filterwarnings("ignore")

import optuna
import numpy as np
from sklearn import metrics
from sklearn.base import clone
from optuna.trial import Trial
import json
import shap
import matplotlib.pyplot as plt

from utils.helper_path import FEATURE_DATA_PATH, RESULTS_PATH, FIGURES_PATH
from utils.helper_functions import set_seed, ECGDataset, prepare_data, get_ml_model, \
    get_data_balance, evaluate_classifier, create_directory, FeatureSelectionPipeline, \
    bootstrap_test_performance, plot_calibration_curve, get_feature_importance_model
from utils.helper_argparse import validate_scaler, validate_category, validate_target_metric, validate_ml_model, \
    validate_resampling_method
from utils.helper_xai import create_shap_dependence_plots, create_shap_beeswarm_plot, create_shap_decision_plot


MODELS_ABBREVIATION_DICT = {
    "lr": "Logistic regression",
    "rf": "Random Forest",
    "dt": "Decision Tree",
    "knn": "K-nearest Neighbor",
    "adaboost": "Adaboost",
    "xgboost": "Extreme Gradient Boosting",
    "lda": "Linear discriminant analysis",
    "qda": "Quadratic discriminant analysis",
    "svm": "Support vector machines",
    "random_baseline": "Random baseline",
}

LABEL_ABBREVIATION_DICT = {
    "mental_stress": "MS",
    "baseline": "BASE",
    "high_physical_activity": "HPA",
    "moderate_physical_activity": "MPA",
    "low_physical_activity": "LPA",
    "rest": "REST"
}


def objective(trial: Trial,
              train_data: tuple,
              val_data: tuple,
              model_type: str,
              metric: str = "roc_auc",
              ) -> float:
    """
    Objective function for Optuna optimization.
    Returns validation balanced accuracy as the optimization metric.
    """

    # Define hyperparameter search space based on model type
    if model_type.lower() == "lr":
        params = {
            'C': trial.suggest_float('C', 0.01, 1, log=True),
            'penalty': "l2",
            'max_iter': 2000,
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            'n_jobs': -1,
        }
    elif model_type.lower() == "rf":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 25),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 25),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            'n_jobs': -1,
        }
    elif model_type.lower() == "xgboost":
        params = {
            # Reduce from 300 to focus on preventing overfitting
            'n_estimators': trial.suggest_int('n_estimators', 25, 200),
            # Reduce max_depth to prevent overfitting
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            # Slower learning rate for better generalization
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),

            # Increase minimum values for stronger regularization
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),

            # Increase regularization range
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0),

            'use_label_encoder': False,
            'n_jobs': -1
        }
    elif model_type.lower() == "dt":
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
        }
    elif model_type.lower() == "adaboost":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'algorithm': trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R'])
        }
    elif model_type.lower() == "knn":
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 50),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 2),  # 1 for manhattan_distance, 2 for euclidean_distance
            'leaf_size': trial.suggest_int('leaf_size', 20, 50),
            "n_jobs": -1
        }
    elif model_type.lower() == "lda":
        params = {
            'solver': trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen']),
            'shrinkage': trial.suggest_float('shrinkage', 0.0, 1.0) if trial.suggest_categorical('use_shrinkage', [True,False]) else None,
            'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True)
        }
    elif model_type.lower() == "qda":
        params = {
            'reg_param': trial.suggest_float('reg_param', 0.0, 1.0),
            'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True)
        }

    elif model_type.lower() == "svm":
        params = {
            "C": trial.suggest_float("C", 0.0, 5.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }

    elif model_type.lower() == "random_baseline":
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


def load_best_params(file_path: str, file_name:str) -> dict[str, Any]:
    try:
        with open(os.path.join(file_path, file_name)) as file:
            print(f"We found optimized parameter configurations and load it")
            return json.load(file)["best_params"]
    except FileNotFoundError:
        return None

def get_save_name(study_name: str, add_within_comparison: bool,
                  use_default_values: bool, use_feature_selection: bool, bootstrap: bool) -> str:
    """Generate the save filename based on the configuration.
    
    Args:
        study_name: Name of the study
        add_within_comparison: Boolean. If set, we run a within-study comparison
        use_default_values: Boolean. If set, we do not do hyperparameter tuning and use the default values
        use_feature_selection: Boolean. If set, we use feature selection
        bootstrap. Boolean. If set, we use bootstrap
    
    Returns:
        str: Filename for saving results
    """
    # Start with base name
    prefix = "WITHIN_" if add_within_comparison else ""
    middle = "DEFAULT_" if use_default_values else ""
    suffix = "_feature_selection" if use_feature_selection else ""
    end = "_boootstrapped" if bootstrap else ""
    
    return f"{prefix}{middle}{study_name}_best_performance_results{suffix}{end}.json"

def optimize_hyperparameters(
    model_type: str,
    train_data: tuple,
    val_data: tuple,
    study_name: str,
    results_path_history: str,
    use_default_values: bool = False,
    do_within_comparison: bool = False,
    do_hyperparameter_tuning: bool = False,
    n_trials: int = 5,
    timeout: int = 3600,
    metric_to_optimize: str = "roc_auc",
    seed: int = 42,
) -> dict[str, Any]:
    """Handle hyperparameter optimization logic for model training.
    
    Args:
        model_type: Type of model to optimize
        train_data: Tuple of (X_train, y_train)
        val_data: Tuple of (X_val, y_val)
        study_name: Name for the optimization study
        results_path_history: Path to save/load optimization history
        use_default_values: If True, use default hyperparameters
        do_within_comparison: If true, then no hyperparameter tuning is done as we do not have a validation set
        do_hyperparameter_tuning: If True, perform hyperparameter optimization
        n_trials: Number of optimization trials
        timeout: Maximum optimization time in seconds
        metric_to_optimize: Metric to optimize for
        seed: Random seed for reproducibility
    
    Returns:
        dict: Best hyperparameters for the model
    """
    if do_hyperparameter_tuning and not do_within_comparison:
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=seed)
        )

        study.optimize(
            lambda trial: objective(trial, train_data, val_data, model_type, metric_to_optimize),
            n_trials=n_trials,
            timeout=timeout
        )
        
        # We will save the best results now here:
                # Save study statistics and best parameters
        study_stats = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "study_name": study_name,
            "model_type": model_type,
            "optimization_history": [
                {"number": t.number, "value": t.value, "params": t.params}
                for t in study.trials
            ]
        }

        save_name = f"{study_name}_optimization_history_feature_selection.json" \
            if args.use_feature_selection else f"{study_name}_optimization_history.json"
        with open(os.path.join(results_path_history, save_name), "w") as f:
            json.dump(study_stats, f, indent=4)

        return study.best_params    

    if use_default_values:
        print("We use the default hyperparameter values")
        return {}

    print(f"We load the best parameter set")
    return load_best_params(results_path_history, f"{study_name}_optimization_history.json")


def main(args):
    target_data_path = os.path.join(FEATURE_DATA_PATH, str(args.sample_frequency), str(args.window_size),
                                    str(args.window_shift))

    # Create path folder depending on the comparison we are trying to do
    comparison = f"{LABEL_ABBREVIATION_DICT[args.positive_class]}_{LABEL_ABBREVIATION_DICT[args.negative_class]}"

    results_path_root = os.path.join(RESULTS_PATH, str(args.sample_frequency), str(args.window_size), comparison,
                                args.model_type.lower())

    if args.verbose:
        print(f"We fit the model {MODELS_ABBREVIATION_DICT[args.model_type.lower()]}")

    # Get separate folders for best run and optimization history for better overview and selected features if selected
    results_path_best_performance = os.path.join(results_path_root, "best_performance")
    results_path_history = os.path.join(results_path_root, "history")
    results_path_feature_selection = os.path.join(results_path_root, "feature_selection")
    results_path_bootstrap_performance = os.path.join(results_path_root, "bootstrap_test")

    # Figures path
    figures_path_root = os.path.join(FIGURES_PATH, str(args.sample_frequency), str(args.window_size), comparison,
                                     args.model_type.lower())

    create_directory(results_path_best_performance)
    create_directory(results_path_history)
    create_directory(results_path_feature_selection)
    create_directory(results_path_bootstrap_performance)
    create_directory(figures_path_root)
    ecg_dataset = ECGDataset(target_data_path, add_participant_id=args.do_within_comparison)

    if args.use_feature_selection:
        # Get the dataset for the feature selection process (we should test it on the test set, to see how it generalizes)
        train_data_feature_selection, val_data_feature_selection = ecg_dataset.get_feature_selection_data()

        train_data_feature_selection, val_data_feature_selection, feature_names = prepare_data(
            train_data_feature_selection,
            val_data_feature_selection,
            positive_class=args.positive_class,
            negative_class=args.negative_class,
            resampling_method=args.resampling_method,
            scaler=args.standard_scaler
        )

        # Create base estimator
        base_estimator = get_ml_model(args.model_type, {})  # Basic model for feature selection

        # Initialize feature selection pipeline
        feature_selector = FeatureSelectionPipeline(
            base_estimator=base_estimator,
            n_features_range=range(args.min_features, args.max_features + 1),
            n_splits=args.n_splits,
            n_trials=10,
            scoring=args.metric_to_optimize,
            random_state=args.seed
        )

        feature_selector.fit(train_data_feature_selection, val_data_feature_selection,
                             feature_names=feature_names,
                             save_path=results_path_feature_selection)

        # best_feature_mask has 104 features
        selected_features = list(feature_selector.best_features_mask)

    if args.do_within_comparison:
        train_data, test_data = ecg_dataset.train_data_within, ecg_dataset.test_data_within
        val_data = None
    else:
        # Get the regular datasplit for the normal between people split
        train_data, val_data, test_data = ecg_dataset.get_data()

    train_data, val_data, test_data, feature_names = prepare_data(
        train_data,
        val_data,
        test_data,
        positive_class=args.positive_class,
        negative_class=args.negative_class,
        resampling_method=args.resampling_method,
        scaler=args.standard_scaler,
        use_subset=selected_features if args.use_feature_selection else None,
    )

    # Setup for hyperparameter optimization
    study_name = f"{args.resampling_method}_{args.model_type.lower()}"
    
    best_params = optimize_hyperparameters(
        model_type=args.model_type,
        train_data=train_data,
        val_data=val_data,
        study_name=study_name,
        results_path_history=results_path_history,
        use_default_values=args.use_default_values,
        do_within_comparison=args.do_within_comparison,
        do_hyperparameter_tuning=args.do_hyperparameter_tuning,
        n_trials=args.n_trials,
        timeout=args.timeout,
        metric_to_optimize=args.metric_to_optimize,
        seed=args.seed
    )

    best_model = get_ml_model(args.model_type, best_params)
    best_model.fit(train_data[0], train_data[1])

    # Evaluate final model
    evaluate_classifier(
        best_model, train_data, val_data, test_data,
        save_path=results_path_best_performance,
        save_name=get_save_name(
            study_name, add_within_comparison=args.do_within_comparison,
            use_default_values=args.use_default_values, 
            use_feature_selection=args.use_feature_selection,
            bootstrap=False
        ),
        verbose=args.verbose)

    if args.bootstrap_test_results:
        final_bootstrapped_results = bootstrap_test_performance(
            best_model,
            test_data,
            args.bootstrap_samples,
            args.bootstrap_method,
        )
        if args.verbose:
            print(final_bootstrapped_results)

        save_name=get_save_name(
            study_name, add_within_comparison=args.do_within_comparison,
            use_default_values=args.use_default_values,
            use_feature_selection=args.use_feature_selection,
            bootstrap=True
        )
        with open(os.path.join(results_path_bootstrap_performance, save_name), "w") as f:
            json.dump(final_bootstrapped_results, f, indent=4)

    if args.add_calibration_plots and not args.do_within_comparison and not args.use_default_values:
        # Get class 1 probability
        y_pred = best_model.predict_proba(test_data[0])[:, 1]
        plot_calibration_curve(test_data[1], y_pred, args.bin_size, args.bin_strategy, figures_path_root)

    # Get the feature importance:
    # Getting the feature names in a better format
    feature_names = [name.replace("_", " ") for name in feature_names]

    # Feature coefficients for LR model
    # print(get_feature_importance_model(best_model, feature_names)[:10])

    # XAI now only for between person
    if args.model_type != "rf" and not args.do_within_comparison and not args.use_default_values:
        explainer = shap.Explainer(best_model, train_data[0], feature_names=feature_names)
        _, shap_values = create_shap_beeswarm_plot(explainer, test_data[0], figures_path=figures_path_root, study_name=study_name,
                                  feature_selection=args.use_feature_selection, max_display=11)
        create_shap_dependence_plots(shap_values, feature_names, figures_path=figures_path_root,
                                     study_name=study_name, feature_selection=args.use_feature_selection,
                                     n_top_features=5)

        create_shap_decision_plot(
            best_model,
            explainer,
            test_data,
            feature_names,
            prediction_filter='correct',
            confidence_threshold=0.9,
            figures_path=figures_path_root,
            study_name=study_name,
            feature_selection=args.use_feature_selection
        )

        create_shap_decision_plot(
            best_model,
            explainer,
            test_data,
            feature_names,
            prediction_filter='incorrect',
            confidence_threshold=0.9,
            figures_path=figures_path_root,
            study_name=study_name,
            feature_selection=args.use_feature_selection
        )


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
    parser.add_argument('--window_shift', type=float, default=10,
                        help="The window shift that we use for detecting stress")
    parser.add_argument("--model_type", help="which model to use"
                                             "Choose from: 'dt', 'rf', 'adaboost', 'lda', "
                                             "'knn', 'lr', 'xgboost', 'qda', 'svm'",
                        type=validate_ml_model, default="lr")
    parser.add_argument("--resampling_method", help="what resampling technique should be used. "
                                                 "Options: 'downsample', 'upsample', 'smote', 'adasyn', 'None'",
                        type=validate_resampling_method, default=None)
    parser.add_argument("--verbose", help="Verbose output", action="store_true")
    parser.add_argument("--use_default_values", action="store_true", help="if set, we do not do hyperparameter tuning and use the default values")
    parser.add_argument("--do_hyperparameter_tuning", action="store_true", help="if set, we do hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of optimization trials for Optuna")
    parser.add_argument("--metric_to_optimize", type=validate_target_metric, default="roc_auc")
    parser.add_argument("--bootstrap_test_results", action="store_true",
                        help="if set, we use bootstrapping to get uncertainty estimates of the test performance.")
    parser.add_argument("--bootstrap_samples", help="number of bootstrap samples.",
                        default=200, type=int)
    parser.add_argument("--bootstrap_method", help="which bootstrap method to use. Options: 'quantile', 'BCa', 'se'",
                        default="quantile")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout for optimization in seconds")
    parser.add_argument("--use_feature_selection", action="store_true", help="Boolean. If set, we use feature selection")
    parser.add_argument("--min_features", type=int, default=5,
                       help="Minimum number of features to select")
    parser.add_argument("--max_features", type=int, default=25,
                       help="Maximum number of features to select")
    parser.add_argument("--n_splits", help="Number of splits used for feature selection.", type=int, default=5)

    parser.add_argument("--add_calibration_plots", action="store_true", help="If set, we will plot calibration plots")
    parser.add_argument("--do_within_comparison", action="store_true", help="If set, we will run a within study for comparison reasons.")

    parser.add_argument("--bin_size", help="what bin size to use for plotting the calibration plots",
                        default=10, type=int)
    parser.add_argument("--bin_strategy", help="what binning strategy to use",
                        default="uniform", choices=("uniform", "quantile")
                        )
    args = parser.parse_args()

    args.verbose = True

    # Set seed for reproducibility
    set_seed(args.seed)

    main(args)

    #ToDo:
    # Add similarity DTW time-series, check how fast this is

    # Useful discussion for the choice of evaluation metrics:
    # See link: https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc

    #ToDo:
    # Get the right transformation for each of the features (min-max scaling, log transform if data is heavily skewed)
    # Feature selection:
    # First do some initial hyperparameter on the val set (for lets say 10 trials) via Bayesian Hyperparameter tuning
    # Save the best configs and then we can load it
    # Run then the finetuning on the selected features
    # take the best config,
    # And then do the feature selection on it)
    # Finetune

