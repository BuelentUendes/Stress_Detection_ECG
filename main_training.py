# Simple script to train machine learning models on the stress dataset

import os
import argparse
from typing import Any
import warnings
warnings.filterwarnings("ignore")

import optuna
from sklearn import metrics
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
            'n_estimators': trial.suggest_int('n_estimators', 25, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1.0, log=True),
            'objective': 'binary:logistic',

            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),

            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0),
            
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
            print(f"We found the .json file and load it")
            return json.load(file)["best_params"]
    except FileNotFoundError:
        return None


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

    ecg_dataset = ECGDataset(target_data_path)

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

    print(selected_features)
    # Get the regular datasplit
    train_data, val_data, test_data = ecg_dataset.get_data()

    # Get the histogram
    # ecg_dataset.plot_histogram("hr_mean", "Mean Heart Rate", save_path=FIGURES_PATH)

    # ToDo:
    # Add a section of only subset of features

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

    # Get the data balance
    data_balance = get_data_balance(train_data[1], val_data[1], test_data[1])

    # Setup for hyperparameter optimization
    study_name = f"{args.resampling_method}_{args.model_type.lower()}"

    if args.do_hyperparameter_tuning:
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=args.seed)
        )

        # Run optimization
        study.optimize(
            lambda trial: objective(trial, train_data, val_data, args.model_type, args.metric_to_optimize),
            n_trials=args.n_trials,
            timeout=args.timeout  # in seconds
        )

        # Get best parameters and train final model
        best_params = study.best_params

    else:
        best_params = load_best_params(results_path_history, f"{study_name}_optimization_history.json")

    best_model = get_ml_model(args.model_type, best_params)
    best_model.fit(train_data[0], train_data[1])

    # Evaluate final model
    evaluate_classifier(
        best_model, train_data, val_data, test_data,
        save_path=results_path_best_performance,
        save_name=f"{study_name}_best_performance_results_feature_selection.json" if args.use_feature_selection else f"{study_name}_best_performance_results.json",
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

        save_name = f"{study_name}_bootstrapped_feature_selection.json" if args.use_feature_selection else f"{study_name}_bootstrapped.json"
        with open(os.path.join(results_path_bootstrap_performance, save_name), "w") as f:
            json.dump(final_bootstrapped_results, f, indent=4)

    if args.do_hyperparameter_tuning:
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

        save_name = f"{study_name}_optimization_history_feature_selection.json" \
            if args.use_feature_selection else f"{study_name}_optimization_history.json"
        with open(os.path.join(results_path_history, save_name), "w") as f:
            json.dump(study_stats, f, indent=4)

    if args.add_calibration_plots:
        # Get class 1 probability
        y_pred = best_model.predict_proba(test_data[0])[:, 1]
        plot_calibration_curve(test_data[1], y_pred, args.bin_size, args.bin_strategy, figures_path_root)

    # Get the feature importance:
    # Getting the feature names in a better format
    feature_names = [name.replace("_", " ") for name in feature_names]

    get_feature_importance_model(best_model, feature_names)

    # Get the shap values
    explainer = shap.Explainer(best_model, train_data[0], feature_names=feature_names)
    print(f"We are getting the explanations")
    shap_values = explainer(test_data[0])

    # We can also do the prediction paths to see the predictions that are highly predictive then for ranging 0.95 - 1.
    # For mental stress vs baseline, mental stress vs

    save_name_shap = f"{study_name}_shap_beeswarm_feature_selection.png" if args.use_feature_selection else \
        f"{study_name}_shap_beeswarm.png"
    # Create and save the beeswarm plot
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, show=False, max_display=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path_root, f"{save_name_shap}"),
                dpi=500,
                bbox_inches='tight')
    plt.close()


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
                        default=512, type=int)
    parser.add_argument("--window_size", type=int, default=60, help="The window size that we use for detecting stress")
    parser.add_argument('--window_shift', type=int, default=10,
                        help="The window shift that we use for detecting stress")
    parser.add_argument("--model_type", help="which model to use"
                                             "Choose from: 'dt', 'rf', 'adaboost', 'lda', "
                                             "'knn', 'lr', 'xgboost', 'qda', 'svm'",
                        type=validate_ml_model, default="lr")
    parser.add_argument("--resampling_method", help="what resampling technique should be used. "
                                                 "Options: 'downsample', 'upsample', 'smote', 'adasyn', 'None'",
                        type=validate_resampling_method, default=None)
    parser.add_argument("--verbose", help="Verbose output", action="store_true")
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
    parser.add_argument("--min_features", type=int, default=91,
                       help="Minimum number of features to select")
    parser.add_argument("--max_features", type=int, default=95,
                       help="Maximum number of features to select")
    parser.add_argument("--n_splits", help="Number of splits used for feature selection.", type=int, default=5)

    parser.add_argument("--add_calibration_plots", action="store_true", help="If set, we will plot calibration plots")
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