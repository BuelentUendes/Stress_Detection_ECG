# Simple script to train machine learning models on the stress dataset

import os
import argparse
import pickle
from typing import Any
import warnings

warnings.filterwarnings("ignore")

import optuna
import numpy as np
import itertools
from sklearn import metrics
from sklearn.base import BaseEstimator
from optuna.trial import Trial
import json
import shap
import pandas as pd

from utils.helper_path import FEATURE_DATA_PATH, RESULTS_PATH, FIGURES_PATH
from utils.helper_functions import (set_seed, ECGDataset, prepare_data, get_ml_model, \
    get_data_balance, evaluate_classifier, create_directory, FeatureSelectionPipeline, \
    bootstrap_test_performance, plot_calibration_curve, get_feature_importance_model, plot_feature_importance,
                                    get_bootstrapped_cohens_kappa, get_bootstrapped_brier_score,
                                    calibrate_isotonic_regression)
from utils.helper_argparse import validate_scaler, validate_category, validate_target_metric, validate_ml_model, \
    validate_resampling_method, validate_feature_subset
from utils.helper_xai import (create_shap_dependence_plots, create_shap_beeswarm_plot,
                              create_shap_decision_plot, create_shap_summary_plot_simple, create_feature_name_dict)


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
    "gmm": "Gaussian Mixture",
}

LABEL_ABBREVIATION_DICT = {
    "mental_stress": "MS",
    "baseline": "BASE",
    "high_physical_activity": "HPA",
    "moderate_physical_activity": "MPA",
    "low_physical_activity": "LPA",
    "rest": "REST",
    "any_physical_activity": "ANY_PHY",
    "non_physical_activity": "NON_PHY",
    "standing": "STANDING",
    "walking_own_pace": "WALKING",
    "low_moderate_physical_activity": "LP_MPA",
    "base_lpa_mpa": "BASE_LPA_MPA",
    "ssst": "SSST",
    "raven": "RAVEN",
    "ta": "TA",
    "pasat": "PASAT",
    "pasat_repeat": "PASAT_REPEAT",
    "ta_repeat": "TA_REPEAT",
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
            'n_estimators': trial.suggest_int('n_estimators', 75, 150),
            # Reduce max_depth to prevent overfitting
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            # lower learning rate for better generalization
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),

            'subsample': trial.suggest_float('subsample', 0.3, 0.6),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),

            # Increase regularization range
            'reg_lambda': trial.suggest_float('reg_lambda', 15., 25.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 15., 25.0),

            'use_label_encoder': False,
            'n_jobs': -1
        }

    elif model_type.lower() == "dt":
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 7),
            'min_samples_split': trial.suggest_int('min_samples_split', 15, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'criterion': trial.suggest_categorical('criterion', ['entropy']),
            # Add max_features to consider fewer features at each split
            'max_features': trial.suggest_float('max_features', 0.6, 0.8),
            'class_weight': trial.suggest_categorical('class_weight', [None])
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
            "probability": True,
            "kernel": "rbf",
        }

    elif model_type.lower() == "random_baseline":
        params = {
            "strategy": "most_frequent"
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
    print(f"we found the model in {file_path}, {file_name}")
    try:
        with open(os.path.join(file_path, file_name)) as file:
            print(f"We found optimized parameter configurations and load it")
            return json.load(file)["best_params"]
    except FileNotFoundError:
        return None


def load_best_model(file_path: str, file_name:str) -> BaseEstimator:
    try:
        with open(os.path.join(file_path, f"{file_name}.pkl"), "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None


def load_top_features(file_path: str, file_name:str, top_k: int = 5,  random: bool=False,
                      random_k: int = 20) -> list[str]:
    try:
        with open(os.path.join(file_path, file_name)) as file:
            print(f"We found optimized parameter configurations and load it")
            feature_selection_results = json.load(file)

            # Now pick either features that are selected up to threshold or random
            if not random:
                feature_names = []
                feature_idx = 0

                while len(feature_names) < top_k:
                    feature_names.append(feature_selection_results[feature_idx][0])
                    feature_idx += 1

                # selected_features_pairs = [feature for feature in feature_selection_results if feature[1] >= threshold]
                # # We need a list of features only
                # feature_names = [feature[0] for feature in selected_features_pairs if feature[1] >= threshold]
            else:
                print(f"we use a random subset of {random_k} features")
                feature_names_list = [feature[0] for feature in feature_selection_results]
                # Ensure that we do not choose more features that we actually have
                number_of_random_samples = min(len(feature_names_list), random_k)
                feature_names = np.random.choice(feature_names_list, size=number_of_random_samples, replace=False)

            return feature_names

    except FileNotFoundError:
        return None


def get_save_name(study_name: str,
                  add_within_comparison: bool,
                  use_default_values: bool,
                  use_feature_selection: bool,
                  use_feature_subset: bool,
                  feature_subset: list[str],
                  top_k_features: int,
                  bootstrap: bool,
                  subcategories: bool,
                  random_subset: bool = False,
                  use_top_features: bool = False,
                  ) -> str:
    """Generate the save filename based on the configuration.
    
    Args:
        study_name: Name of the study
        add_within_comparison: Boolean. If set, we run a within-study comparison
        use_default_values: Boolean. If set, we do not do hyperparameter tuning and use the default values
        use_feature_selection: Boolean. If set, we use feature selection
        bootstrap. Boolean. If set, we use bootstrap
        subcategories. Boolean. If set, we bootstrapped subcategories
        random_subset. Boolean. If set, we used a random subset of features
    
    Returns:
        str: Filename for saving results
    """
    # Start with base name
    prefix = "WITHIN_" if add_within_comparison else ""
    middle = "DEFAULT_" if use_default_values else ""
    suffix = "_feature_selection" if use_feature_selection else ""
    if use_feature_subset and not random_subset:
        suffix_2 = f"_subset_features_{len(feature_subset)}" if use_feature_subset else ""

    elif use_top_features:
        suffix_2 = f"_subset_features_top_{str(top_k_features)}"

    elif use_feature_subset and random_subset:
        suffix_2 = "_subset_features_random"

    else:
        suffix_2 = ""

    if bootstrap:
        end = "_bootstrapped_subcategories" if subcategories else "_bootstrapped"
    else:
        end = ""

    return f"{prefix}{middle}{study_name}_best_performance_results{suffix}{end}.json" if not bootstrap else \
        f"{prefix}{middle}{study_name}{suffix}{suffix_2}{end}.json"

def optimize_hyperparameters(
    model_type: str,
    train_data: tuple,
    val_data: tuple,
    study_name: str,
    results_path_history: str,
    use_default_values: bool = True,
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

    elif use_default_values:
        print("We use the default hyperparameter values")
        return {}

    else:
        print(f"We load the best parameter set")
        return load_best_params(results_path_history, f"{study_name}_optimization_history.json")


def get_subset_feature_df(df: pd.DataFrame, feature_subset: list[str]) -> pd.DataFrame:
    # We need to have anyways the label and category in there
    feature_list = ["category", "label"]
    feature_list.extend(feature for feature in feature_subset)
    return df[feature_list]


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
    if args.leave_one_out:
        results_path_best_performance = os.path.join(results_path_root, "leave_one_out", args.leave_out_stressor_name,
                                                     "best_performance")
        results_path_history = os.path.join(results_path_root, "leave_one_out", args.leave_out_stressor_name,
                                            "history")
        results_path_feature_selection = os.path.join(results_path_root, "leave_one_out", args.leave_out_stressor_name,
                                                      "feature_selection")
        results_path_bootstrap_performance = os.path.join(results_path_root, "leave_one_out", args.leave_out_stressor_name,
                                                          "bootstrap_test")
        results_path_model_weights = os.path.join(results_path_root, "leave_one_out", args.leave_out_stressor_name,
                                                  "best_model_weights")

    elif args.balance_positive_sublabels:
        results_path_best_performance = os.path.join(results_path_root, "subsample", args.balance_sublabels_method,
                                                     "best_performance")
        results_path_history = os.path.join(results_path_root, "subsample", args.balance_sublabels_method,
                                            "history")
        results_path_feature_selection = os.path.join(results_path_root, "subsample", args.balance_sublabels_method,
                                                      "feature_selection")
        results_path_bootstrap_performance = os.path.join(results_path_root, "subsample", args.balance_sublabels_method,
                                                          "bootstrap_test")
        results_path_model_weights = os.path.join(results_path_root, "subsample", args.balance_sublabels_method,
                                                  "best_model_weights")
        results_path_bootstrap_train_performance = os.path.join(results_path_root, "bootstrap_train")
        results_path_bootstrap_val_performance = os.path.join(results_path_root, "bootstrap_val")

    else:
        results_path_best_performance = os.path.join(results_path_root, "best_performance")
        results_path_history = os.path.join(results_path_root, "history")
        results_path_feature_selection = os.path.join(results_path_root, "feature_selection")
        results_path_bootstrap_performance = os.path.join(results_path_root, "bootstrap_test")
        results_path_bootstrap_train_performance = os.path.join(results_path_root, "bootstrap_train")
        results_path_bootstrap_val_performance = os.path.join(results_path_root, "bootstrap_val")
        results_path_model_weights = os.path.join(results_path_root, "best_model_weights")

    # Figures path
    figures_path_hist = os.path.join(FIGURES_PATH, str(args.sample_frequency), str(args.window_size), comparison)
    figures_path_feature_plots = os.path.join(FIGURES_PATH, str(args.sample_frequency), str(args.window_size), "feature_plots")
    figures_path_root = os.path.join(FIGURES_PATH, str(args.sample_frequency), str(args.window_size), comparison,
                                     args.model_type.lower())

    create_directory(results_path_best_performance)
    create_directory(results_path_history)
    create_directory(results_path_feature_selection)
    create_directory(results_path_bootstrap_performance)
    if not args.leave_one_out:
        create_directory(results_path_bootstrap_train_performance)
        create_directory(results_path_bootstrap_val_performance)

    create_directory(figures_path_root)
    create_directory(figures_path_feature_plots)
    create_directory(results_path_model_weights)

    ecg_dataset = ECGDataset(target_data_path, add_participant_id=args.do_within_comparison)

    if args.negative_class in ["baseline", "low_physical_activity"] and args.positive_class not in ["low_physical_activity", "moderate_physical_activity", "high_physical_activity", "any_physical_activity"]:
        reference = "Sitting" if args.negative_class == "baseline" else "Standing"
        # We plot the reference HR reactivity only always against the true negative reference class sitting
        ecg_dataset.get_average_hr_reactivity_box(args.positive_class, args.negative_class, save_path=figures_path_hist,
                                                  reference=reference, show_plot=False)

    ecg_dataset.plot_histogram(column="hr_mean", x_label="Mean Heart Rate (bpm)",
                               show_baseline=False,
                               save_path=figures_path_hist, show_plot=False)

    if args.use_feature_selection:
        # Get the dataset for the feature selection process (we should test it on the test set, to see how it generalizes)
        train_data_feature_selection, val_data_feature_selection = ecg_dataset.get_feature_selection_data()

        train_data_feature_selection, val_data_feature_selection, feature_names = prepare_data(
            train_data_feature_selection, val_data_feature_selection, positive_class=args.positive_class,
            negative_class=args.negative_class, resampling_method=args.resampling_method,
            balance_positive_sublabels=args.balance_positive_sublabels,
            balance_sublabels_method = args.balance_sublabels_method,
            scaler=args.standard_scaler,
            use_quantile_transformer=args.use_quantile_transformer)

        # Create base estimator
        base_estimator = get_ml_model(args.model_type, {})  # Basic model for feature selection

        # Initialize feature selection pipeline
        feature_selector = FeatureSelectionPipeline(
            base_estimator=base_estimator,
            feature_names= feature_names,
            n_features_range=range(args.min_features, args.max_features + 1),
            n_splits=args.n_splits,
            n_trials=10,
            scoring=args.metric_to_optimize,
            random_state=args.seed
        )

        feature_selector.fit(train_data_feature_selection, val_data_feature_selection,
                             feature_names=feature_names,
                             top_k_features=args.top_k_features,
                             save_path=results_path_feature_selection)

        # best_feature_mask has 104 features
        selected_features = list(feature_selector.best_features_mask)

    if args.do_within_comparison:
        train_data, test_data = ecg_dataset.train_data_within, ecg_dataset.test_data_within
        val_data = None
    else:
        # Get the regular datasplit for the normal between people split
        train_data, val_data, test_data = ecg_dataset.get_data()

    if args.use_feature_subset:
        # We need to use here a function if args.use_top_features
        if args.use_top_features:
            args.feature_subset = load_top_features(
                file_path=results_path_feature_selection,
                file_name="feature_importance_total_selected.json",
                top_k=args.top_k_features,
                random=args.use_random_subset_features,
                random_k=10
            )

        train_data = get_subset_feature_df(train_data, feature_subset=args.feature_subset)
        val_data = get_subset_feature_df(val_data, feature_subset=args.feature_subset)
        test_data = get_subset_feature_df(test_data, feature_subset=args.feature_subset)

    train_data, val_data, test_data, feature_names = prepare_data(train_data, val_data, test_data,
                                                                  positive_class=args.positive_class,
                                                                  negative_class=args.negative_class,
                                                                  resampling_method=args.resampling_method,
                                                                  balance_positive_sublabels=args.balance_positive_sublabels,
                                                                  balance_sublabels_method=args.balance_sublabels_method,
                                                                  scaler=args.standard_scaler,
                                                                  use_quantile_transformer=args.use_quantile_transformer,
                                                                  use_subset=selected_features if args.use_feature_selection else None,
                                                                  save_path=figures_path_feature_plots,
                                                                  save_feature_plots=args.save_feature_plots,
                                                                  leave_one_out=args.leave_one_out,
                                                                  leave_out_stressor_name=args.leave_out_stressor_name)

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
    evaluation_results = evaluate_classifier(
        best_model, train_data, val_data, test_data,
        save_path=results_path_best_performance,
        save_name=get_save_name(
            study_name,
            add_within_comparison=args.do_within_comparison,
            use_default_values=args.use_default_values, 
            use_feature_selection=args.use_feature_selection,
            use_feature_subset=args.use_feature_subset,
            feature_subset=args.feature_subset,
            top_k_features=args.top_k_features,
            use_top_features=args.use_top_features,
            bootstrap=False,
            subcategories=False,
            random_subset=args.use_random_subset_features,
        ),
        verbose=args.verbose)


    # Save model weights and threshold for classification so I can later retrieve it for cohen's kappa calculation
    # Save the model weights so we can later load them for cohen's kappa
    with open(os.path.join(results_path_model_weights, f"best_model_weights_{args.resampling_method}.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    with open(os.path.join(results_path_model_weights, f"classification_threshold_{args.resampling_method}.json"), "w") as f:
        classification_threshold = {"classification_threshold f1": evaluation_results["f1_score_threshold"],
                                    "classification_threshold precision:": evaluation_results["precision_score_threshold"],
                                    "classification_threshold recall": evaluation_results["recall_score_threshold"]}

        json.dump(classification_threshold, f)

    if args.bootstrap_test_results:
        set_seed(args.seed)
        final_bootstrapped_results = bootstrap_test_performance(
            best_model,
            test_data,
            args.bootstrap_samples,
            args.bootstrap_method,
            evaluation_results["f1_score_threshold"],
            args.bootstrap_subcategories,
            args.leave_one_out,
            args.leave_out_stressor_name,
        )

        #ToDo: Refactor this code:
        if not args.leave_one_out:
            set_seed(args.seed)
            final_bootstrapped_results_train = bootstrap_test_performance(
                best_model,
                train_data,
                args.bootstrap_samples,
                args.bootstrap_method,
                evaluation_results["f1_score_threshold"],
                False,
                args.leave_one_out,
                args.leave_out_stressor_name,
            )

            set_seed(args.seed)
            final_bootstrapped_results_val = bootstrap_test_performance(
                best_model,
                val_data,
                args.bootstrap_samples,
                args.bootstrap_method,
                evaluation_results["f1_score_threshold"],
                False,
                args.leave_one_out,
                args.leave_out_stressor_name,
            )

        if args.verbose:
            print(final_bootstrapped_results[0])

        save_name_overall=get_save_name(
            study_name,
            add_within_comparison=args.do_within_comparison,
            use_default_values=args.use_default_values,
            use_feature_selection=args.use_feature_selection,
            use_feature_subset=args.use_feature_subset,
            feature_subset = args.feature_subset,
            top_k_features=args.top_k_features,
            use_top_features=args.use_top_features,
            bootstrap=True,
            subcategories=False,
            random_subset=args.use_random_subset_features,
        )

        with open(os.path.join(results_path_bootstrap_performance, save_name_overall), "w") as f:
            json.dump(final_bootstrapped_results[0], f, indent=4)

        if not args.leave_one_out:
            with open(os.path.join(results_path_bootstrap_train_performance, save_name_overall), "w") as f:
                json.dump(final_bootstrapped_results_train[0], f, indent=4)

            with open(os.path.join(results_path_bootstrap_val_performance, save_name_overall), "w") as f:
                json.dump(final_bootstrapped_results_val[0], f, indent=4)

        if args.bootstrap_subcategories:
            save_name_subcategories = get_save_name(
                study_name,
                add_within_comparison=args.do_within_comparison,
                use_default_values=args.use_default_values,
                use_feature_selection=args.use_feature_selection,
                use_feature_subset=args.use_feature_subset,
                feature_subset=args.feature_subset,
                top_k_features=args.top_k_features,
                use_top_features=args.use_top_features,
                bootstrap=True,
                subcategories=True,
                random_subset=args.use_random_subset_features,
            )

            with open(os.path.join(results_path_bootstrap_performance, save_name_subcategories), "w") as f:
                json.dump(final_bootstrapped_results[1], f, indent=4)

        if args.leave_one_out:
            with open(os.path.join(results_path_bootstrap_performance, f"{save_name_overall}_in_distribution_known_stressors.json"), "w") as f:
                json.dump(final_bootstrapped_results[2], f, indent=4)

    if args.add_calibration_plots and not args.do_within_comparison and not args.use_default_values:
        # Get class 1 probability
        y_pred = best_model.predict_proba(test_data[0])[:, 1]
        isotonic_regressor = calibrate_isotonic_regression(best_model, val_data)

        # Transform the probabilities:
        y_pred = isotonic_regressor.transform(y_pred)

        plot_calibration_curve(test_data[1], y_pred, args.bin_size, args.bin_strategy,
                               resampling_method=args.resampling_method,
                               save_path= figures_path_root)

        # Get bootstrapped ECE results:
        bootstrapped_brier_score_results= get_bootstrapped_brier_score(
            best_model, val_data, test_data, args.bootstrap_samples, args.bootstrap_method
        )

        # Save brier score results:
        with open(os.path.join(results_path_best_performance, f"bootstrapped_brier_scores_{args.resampling_method}.json"), "w") as f:
            json.dump(bootstrapped_brier_score_results, f, indent=4)

    # Feature coefficients for LR model
    if args.model_type == "lr":
        lr_coefficients = get_feature_importance_model(best_model, feature_names)
        prefix = f"{args.resampling_method}_feature_coefficients"
        save_name_feature_coefficients = f"{prefix}_feature_selection.json" if args.use_feature_selection else f"{prefix}.json"
        with open(os.path.join(results_path_best_performance, save_name_feature_coefficients), "w") as f:
            json.dump(lr_coefficients, f, indent=4)

        save_name = os.path.join(figures_path_root, f"{save_name_feature_coefficients}.png")
        feature_names_dict = create_feature_name_dict()
        plot_feature_importance(lr_coefficients, num_features=10, figsize=(8, 6),
                                feature_names_dict=feature_names_dict,
                                save_path=save_name)

    # XAI now only for between person
    if args.get_model_explanations:
        if args.model_type not in ["rf", "random_baseline"] and not args.do_within_comparison and not args.use_default_values:
            explainer = shap.Explainer(best_model, train_data[0], feature_names=feature_names)
            feature_names = create_feature_name_dict()
            _, shap_values = create_shap_summary_plot_simple(
                explainer,
                test_data[0],
                figures_path=figures_path_root,
                study_name=study_name,
                feature_name_dict=feature_names,
                feature_selection=args.use_feature_selection,
                max_display=10)

    if args.get_cohens_kappa:
        model_comparisons = args.model_comparisons.split(",")
        model_dict = {}
        root_path= os.path.join(RESULTS_PATH, str(args.sample_frequency), str(args.window_size), comparison)
        file_name = f"best_model_weights_{args.resampling_method}"
        for model in model_comparisons:
            file_path = os.path.join(root_path, model, "best_model_weights")
            with open(os.path.join(file_path, f"classification_threshold_{args.resampling_method}.json"), "r") as f:
                # We set the best threshold to detect
                classification_threshold = json.load(f)["classification_threshold f1"]
            model_dict[model] = (load_best_model(file_path, file_name), classification_threshold)

        final_cohen_results = {}
        for (ml_model_1, model_threshold_pair_1), (ml_model_2, model_threshold_pair_2) in itertools.combinations(model_dict.items(), 2):
            bootstrapped_cohen = get_bootstrapped_cohens_kappa(
                model_threshold_pair_1[0], model_threshold_pair_1[1],
                model_threshold_pair_2[0], model_threshold_pair_2[1],
                test_data, args.bootstrap_samples, args.bootstrap_method
            )
            final_cohen_results[f"{ml_model_1}_{ml_model_2}"] = bootstrapped_cohen

        print(final_cohen_results)

        # Good paper for comparison of cohens kappa:
        # https: // pmc.ncbi.nlm.nih.gov / articles / PMC3900052 /  # t3-biochem-med-22-3-276-4

        # Save cohen kappa results:
        with open(os.path.join(root_path, f"cohen_kappa_{args.resampling_method}.json"), "w") as f:
            json.dump(final_cohen_results, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed number", default=42, type=int)
    parser.add_argument("--positive_class", help="Which category should be 1",
                        default="mental_stress",
                        type=validate_category)
    parser.add_argument("--negative_class", help="Which category should be 0",
                        default="base_lpa_mpa",
                        type=validate_category)
    parser.add_argument("--standard_scaler", help="Which standard scaler to use. "
                                                  "Choose from 'standard_scaler' or 'min_max'",
                        type=validate_scaler,
                        default="standard_scaler")
    parser.add_argument("--use_quantile_transformer", action="store_true")
    parser.add_argument("--sample_frequency",
                        help="which sample frequency to use for the training",
                        default=1000, type=int)
    parser.add_argument("--window_size", type=int, default=30,
                        help="The window size that we use for detecting stress")
    parser.add_argument('--window_shift', type=str, default='10full',
                        help="The window shift that we use for detecting stress")
    parser.add_argument("--model_type", help="which model to use"
                                             "Choose from: 'dt', 'rf', 'adaboost', 'lda', "
                                             "'knn', 'lr', 'xgboost', 'qda', 'svm', random_baseline', 'gmm'",
                        type=validate_ml_model, default="lr")
    parser.add_argument("--resampling_method", help="what resampling technique should be used. "
                                                 "Options: 'downsample', 'upsample', 'smote', 'adasyn', 'None'",
                        type=validate_resampling_method, default=None)
    parser.add_argument("--verbose", help="Verbose output", action="store_true")
    parser.add_argument("--use_default_values", action="store_true",
                        help="if set, we do not do hyperparameter tuning and use the default values")
    parser.add_argument("--do_hyperparameter_tuning", action="store_true",
                        help="if set, we do hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=25, help="Number of optimization trials for Optuna")
    parser.add_argument("--metric_to_optimize", type=validate_target_metric, default="roc_auc")
    parser.add_argument("--bootstrap_test_results", action="store_true",
                        help="if set, we use bootstrapping to get uncertainty estimates of the test performance.")
    parser.add_argument("--bootstrap_samples", help="number of bootstrap samples.",
                        default=200, type=int) # What happens if I do 1000? samples, so far the analysis is with 200
    parser.add_argument("--bootstrap_method",
                        help="which bootstrap method to use. Options: 'quantile', 'BCa', 'se'",
                        default="quantile")
    parser.add_argument("--bootstrap_subcategories", action="store_true",
                        help="If enabled, we also bootstrap the subcategories and get CI for these.")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout for optimization in seconds")
    parser.add_argument("--use_feature_selection", action="store_true",
                        help="Boolean. If set, we use feature selection")
    parser.add_argument("--min_features", type=int, default=5,
                       help="Minimum number of features to select")
    parser.add_argument("--max_features", type=int, default=50,
                       help="Maximum number of features to select")
    parser.add_argument("--use_feature_subset",
                        help="instead of using full set of features, use only a subset of features "
                             "defined in args.subset",
                        action="store_true")
    parser.add_argument("--feature_subset",
                        help="What feature subset to use. Only used when 'use_feature_subset' is set to true",
                        type=validate_feature_subset,
                        default="w,wmax,nn20,nn50")
    parser.add_argument("--use_top_features",
                        help="If set, we use the top features that were selected 100% of the time during feature selection",
                        action="store_true")
    parser.add_argument("--top_k_features",
                        help="If use top features is set, how many features we want to select.",
                        default=3, type=int)
    parser.add_argument("--use_random_subset_features",
                        help="If set, we use a random subset of features.",
                        action="store_true")
    parser.add_argument("--n_splits", help="Number of splits used for feature selection.",
                        type=int, default=5)

    parser.add_argument("--add_calibration_plots", action="store_true",
                        help="If set, we will plot calibration plots")
    parser.add_argument("--do_within_comparison", action="store_true",
                        help="If set, we will run a within study for comparison reasons.")

    parser.add_argument("--bin_size", help="what bin size to use for plotting the calibration plots",
                        default=10, type=int)
    parser.add_argument("--bin_strategy", help="what binning strategy to use",
                        default="uniform", choices=("uniform", "quantile")
                        )
    parser.add_argument("--get_model_explanations", action="store_true",
                        help="If set, we get model explanations using SHAP")
    parser.add_argument("--get_cohens_kappa", action="store_true",
                        help="If set, we calculate the cohens kappa to get the agreement.")
    parser.add_argument("--model_comparisons", default="lr,xgboost",
                        help="For which models we want to get the cohens kappa scores.")
    parser.add_argument("--save_feature_plots", action="store_true",
                        help="If we want to show the distribution of the feature plots. "
                             "If set, we will save the feature plots. This will take longer though!")
    parser.add_argument("--leave_one_out", action="store_true",
                        help="We will train and validate without a stressor")
    parser.add_argument("--leave_out_stressor_name", help="Which stressor to leave out",
                        choices=("ta", "pasat", "raven", "ssst","none", "ta_repeat", "pasat_repeat"),
                        default=None, type=str)
    parser.add_argument("--balance_positive_sublabels", action="store_true",
                        help="If we want to have equal proportions in the training set of positive label.")
    parser.add_argument("--balance_sublabels_method", choices=("downsample", "upsample", "smote"),
                        help="What method to use for the sublabel balancing.", type=str,
                        default="downsample")

    args = parser.parse_args()

    args.verbose = True
    args.bootstrap_test_results = True
    args.bootstrap_subcategories = True
    args.add_calibration_plots = True
    args.use_feature_subset = False
    # args.use_top_features = True
    args.do_hyperparameter_tuning = True
    args.get_model_explanations = True if args.model_type != "rf" else False
    # args.save_feature_plots = True
    # Set seed for reproducibility

    args.resampling_method = "smote"
    # args.min_features = 3
    # args.max_features = 3
    # args.use_feature_selection = True
    # args.use_top_features = True

    # args.balance_positive_sublabels = True
    set_seed(args.seed)

    main(args)

    # Useful discussion for the choice of evaluation metrics:
    # See link: https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc

    #ToDo:
    # Do physical activity vs baseline -> should be high performant.
    # SSST smote upsampling. Stratified approach (either upsample or downsample)
    # Normally then we should see an improved performance in the SSST as well
    # (as the reason why this performs so low is bc of the small sample size).
    # Do the feature reduction experiments as well.
    # From 104 to 20 to 10
    # Get the explanations for the ones with fewer features
    # Get the random also

    #ToDo:
    # Add similarity DTW time-series, check how fast this is
    # Classify performance MS - LPA and MS -MPA (very similar though) solely on mean heart rate and compare with dummy classifier
    # Logging how many samples are removed
    # Check distribution of the underlying features!
    # Alternative feature selection process -> mutual information, backward selection
    # Combat overfitting of XGboost
    # Check the preprocessing/feature engineering pipeline once more! -> must be somehow ways to improve the performance
    # Log it correctly & assess the quality of the signal!
    # window shift of 30s

    #Combat overfitting XGboost

    #In-depth gaussian mixture model:
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html

    # To get the reduction of features:
    # Use this command:
    # python3 main_training.py --model_type xgboost --use_feature_selection --min_features 20 --max_features 20 --top_k_features 20 --do_hyperparameter_tuning --n_trials 25 --use_top_features