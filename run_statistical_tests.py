# Simple sample script to run statistical test to test performance difference between our ML models
# We bootstrap the pairwise performance difference and check if the performance difference is different from zero
# IMPORTANT: This script should only be run after the models have been trained: main_training.py

import os
import json
import argparse
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
import numpy as np
from scipy import stats


from utils.helper_path import FEATURE_DATA_PATH, RESULTS_PATH
from utils.helper_functions import (set_seed, ECGDataset, prepare_data, get_ml_model, \
    get_resampled_data, get_performance_metric_bootstrapped, get_confidence_interval_mean)
from utils.helper_argparse import validate_scaler, validate_category,  validate_ml_model, \
    validate_resampling_method
from main_training import load_best_params

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

def check_significance(result):
    if (result["ci_lower"] < 0)  and (result["ci_upper"] >=0):
        return "not_significant"
    else:
        return "significant"


def calculate_differences(perf_dict, model1, model2):
    """Calculate difference as model1 - model2 for all metrics"""
    return {
        metric: perf_dict[model1][metric] - perf_dict[model2][metric]
        for metric in perf_dict[model1]
    }


def get_bootstrap_test_results(
        delta_list,
        observed_difference,
        hypothesis="greater",
        seed=42,
        resample_null:bool=False
):
    """
    Compute p-value and z-statistic for bootstrap hypothesis test.

    Test statistic: observed_difference (the actual difference from original data)
    Null hypothesis H0: Δ = 0 (no difference between groups)
    Alternative hypothesis H1: Δ > 0 (greater) or Δ ≠ 0 (unequal)

    Parameters:
    -----------
    delta_list: list of bootstrap differences Δ_i = metric_A − metric_B
    observed_difference: TEST STATISTIC - the actual observed difference from original data
    hypothesis: "greater" (one-sided) or "unequal" (two-sided)
    seed: random seed for reproducibility
    resample_null: If we want to resample from the null, this adds additional noise

    Returns:
    --------
    dict: containing p_val, z_stat, and bootstrap_se
    """
    B = len(delta_list)
    mean_delta = sum(delta_list) / B

    # # Center the bootstrap distribution around zero (null hypothesis)
    # null_dist = [d - mean_delta for d in delta_list]

    # Convert to numpy array for easier indexing
    null_dist = np.array(delta_list)

    # Calculate bootstrap standard deviation from the centered null distribution
    # In bootstrapping, the std is the standard error see here:
    # From Effron's "Introduction to the Bootstrap" pg 47:
    bootstrap_se = np.std(null_dist, ddof=1)  # Use sample standard deviation

    # Calculate z-statistic: (observed - expected_under_H0) / SE
    # Under H0, expected difference is 0
    z_stat = observed_difference / bootstrap_se if bootstrap_se > 0 else np.inf

    # Set up random number generator
    rng = np.random.RandomState(seed)

    if resample_null:
    # Resample from the centered null distribution
        null_dist = rng.choice(null_dist, size=B, replace=True)

    # Calculate p-value based on hypothesis type
    if hypothesis == "greater":
        # H1: Δ > 0 (one-sided test)
        p_val = np.sum(null_dist >= observed_difference) / B
    elif hypothesis == "unequal":
        # H1: Δ ≠ 0 (two-sided test)
        p_val = np.sum(np.abs(null_dist) >= np.abs(observed_difference)) / B
    else:
        raise ValueError("hypothesis must be 'greater' or 'unequal'")

    return {
        'p_val': p_val,
        'z_stat': z_stat,
        'bootstrap_sd': bootstrap_se,
        'test_statistic': observed_difference
    }


def main(args):

    target_data_path = os.path.join(FEATURE_DATA_PATH, str(args.sample_frequency), str(args.window_size),
                                    str(args.window_shift))
    # Create path folder depending on the comparison we are trying to do
    comparison = f"{LABEL_ABBREVIATION_DICT[args.positive_class]}_{LABEL_ABBREVIATION_DICT[args.negative_class]}"
    results_path_root = os.path.join(RESULTS_PATH, str(args.sample_frequency), str(args.window_size), comparison)
    ecg_dataset = ECGDataset(target_data_path)
    # Get the regular datasplit for the normal between people split
    train_data, val_data, test_data = ecg_dataset.get_data()
    train_data, val_data, test_data, feature_names = prepare_data(train_data, val_data, test_data,
                                                                  positive_class=args.positive_class,
                                                                  negative_class=args.negative_class,
                                                                  resampling_method=args.resampling_method,
                                                                  balance_positive_sublabels=args.balance_positive_sublabels,
                                                                  balance_sublabels_method=args.balance_sublabels_method,
                                                                  scaler=args.standard_scaler,
                                                                  use_quantile_transformer=args.use_quantile_transformer,
                                                                  use_subset=None,
                                                                  save_feature_plots=args.save_feature_plots,
                                                                  leave_one_out=args.leave_one_out,
                                                                  leave_out_stressor_name=args.leave_out_stressor_name)


    model_comparisons = args.model_comparisons.split(",")
    model_dict = {}
    f1_score_thresholds = {}
    performance_means = {}
    root_path = os.path.join(RESULTS_PATH, str(args.sample_frequency), str(args.window_size), comparison)
    for model_name in model_comparisons:
        # get the model ( we find the best optimization parameter in the history
        file_path = os.path.join(root_path, model_name, "history")
        file_path_threshold = os.path.join(root_path, model_name, "best_model_weights")
        file_name = f"{args.resampling_method}_{model_name}_optimization_history.json"
        performance_path = os.path.join(root_path, model_name, "bootstrap_test")
        performance_file_name = f"{args.resampling_method}_{model_name}_bootstrapped.json"

        model_weights = load_best_params(file_path, file_name)
        model = get_ml_model(model_name, params=model_weights)
        model.fit(train_data[0], train_data[1])
        model_dict[model_name] = model
        with open(os.path.join(file_path_threshold, f"classification_threshold_{args.resampling_method}.json"), "r") as f:
            # We set the best threshold to detect
            classification_threshold = json.load(f)["classification_threshold f1"]
        with open(os.path.join(performance_path, performance_file_name)) as f:
            performance_dict = json.load(f)
            performance_values = {
                metric: values["mean"] for metric, values in performance_dict.items()
            }
            performance_means[model_name] = performance_values
        f1_score_thresholds[model_name] = classification_threshold


    X_test, y_test, label_test = test_data
    results = {
        'roc_auc': [],
        'pr_auc': [],
        'balanced_accuracy': [],
        'f1_score': [],
    }

    # For p value calculation we need to center the distribution
    results_bootstrapped_p_value = {
        'roc_auc': [],
        'pr_auc': [],
    }

    mean_difference_to_check = calculate_differences(performance_means, args.model_comparisons.split(",")[0],
                                                     args.model_comparisons.split(",")[1])

    for idx in tqdm(range(args.bootstrap_samples), desc="Bootstrapping", unit="it"):
        X_bootstrap, y_bootstrap = get_resampled_data(X_test, y_test, seed=idx)

        roc_auc_list = []
        pr_auc_list = []
        balanced_accuracy_score_list = []
        f1_score_list = []

        for model_name, model in model_dict.items():
            roc_auc, pr_auc, balanced_accuracy_score, f1_score, accuracy = get_performance_metric_bootstrapped(
                model, X_bootstrap, y_bootstrap, f1_score_thresholds[model_name])

            roc_auc_list.append(roc_auc)
            pr_auc_list.append(pr_auc)
            balanced_accuracy_score_list.append(balanced_accuracy_score)
            f1_score_list.append(f1_score)

        results["roc_auc"].append(roc_auc_list[0] - roc_auc_list[1])
        results["pr_auc"].append(pr_auc_list[0] - pr_auc_list[1])
        results["balanced_accuracy"].append(balanced_accuracy_score_list[0] - balanced_accuracy_score_list[1])
        results["f1_score"].append(f1_score_list[0] - f1_score_list[1])

        bootstrap_diff = roc_auc_list[0] - roc_auc_list[1]
        bootstrap_diff_pr = pr_auc_list[0] - pr_auc_list[1]
        results_bootstrapped_p_value["roc_auc"].append(bootstrap_diff - mean_difference_to_check["roc_auc"])
        results_bootstrapped_p_value["pr_auc"].append(bootstrap_diff_pr - mean_difference_to_check["pr_auc"])

    p_values = {
        metric: get_bootstrap_test_results(diffs, mean_difference_to_check[metric])
        for metric, diffs in results_bootstrapped_p_value.items()
    }

    print("One-sided bootstrap p-values:", p_values)

    final_diff_results_alpha10 = get_confidence_interval_mean(results,
                                                      bootstrap_method=args.bootstrap_method,
                                                      alpha=10)
    final_diff_results_alpha5 = get_confidence_interval_mean(results,
                                                      bootstrap_method=args.bootstrap_method,
                                                      alpha=5.0)
    final_diff_results_alpha1 = get_confidence_interval_mean(results,
                                                      bootstrap_method=args.bootstrap_method,
                                                      alpha=1.0)
    final_diff_results_alpha01 = get_confidence_interval_mean(results,
                                                      bootstrap_method=args.bootstrap_method,
                                                      alpha=0.1)

    # When the confidence interval excludes 0, then it is significant at that level!
    # If not, we check the p-value respectively
    for metric, results_dict in final_diff_results_alpha10.items():
        final_diff_results_alpha10[metric][f"significant_@_10"]  = check_significance(results_dict)
        if metric in p_values.keys():
            final_diff_results_alpha10[metric].update(p_values[metric])

        with open(os.path.join(root_path, f"statistical_test_{args.model_comparisons.replace(',','_')}_alpha_10.json"),
                  "w") as f:
            json.dump(final_diff_results_alpha10, f, indent=4)

    for metric, results_dict in final_diff_results_alpha5.items():
        final_diff_results_alpha5[metric][f"significant_@_5"]  = check_significance(results_dict)
        if metric in p_values.keys():
            final_diff_results_alpha5[metric].update(p_values[metric])
        # final_diff_results_alpha5[metric][f"p_values"] = p_values
        with open(os.path.join(root_path, f"statistical_test_{args.model_comparisons.replace(',', '_')}_alpha_5.json"),
                  "w") as f:
            json.dump(final_diff_results_alpha5, f, indent=4)

    for metric, results_dict in final_diff_results_alpha1.items():
        final_diff_results_alpha1[metric][f"significant_@_1"]  = check_significance(results_dict)
        if metric in p_values.keys():
            final_diff_results_alpha1[metric].update(p_values[metric])
        # final_diff_results_alpha1[metric][f"p_values"] = p_values
        with open(os.path.join(root_path, f"statistical_test_{args.model_comparisons.replace(',', '_')}_alpha_1.json"),
                  "w") as f:
            json.dump(final_diff_results_alpha1, f, indent=4)

    for metric, results_dict in final_diff_results_alpha01.items():
        final_diff_results_alpha01[metric][f"significant_@_01"]  = check_significance(results_dict)
        if metric in p_values.keys():
            final_diff_results_alpha01[metric].update(p_values[metric])
        # final_diff_results_alpha01[metric][f"p_values"] = p_values
        with open(os.path.join(root_path, f"statistical_test_{args.model_comparisons.replace(',', '_')}_alpha_01.json"),
                  "w") as f:
            json.dump(final_diff_results_alpha01, f, indent=4)

    print(f"Performance_difference {args.model_comparisons} frequency: {args.sample_frequency}"
          f"\n Alpha 10: {final_diff_results_alpha10}",
          f"\n Alpha 5: {final_diff_results_alpha5}",
          f"\n Alpha 1: {final_diff_results_alpha1}",
          f"\n Alpha 0.1: {final_diff_results_alpha01}")


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
                        default=125, type=int)
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
                        type=validate_resampling_method, default="smote")
    parser.add_argument("--verbose", help="Verbose output", action="store_true")
    parser.add_argument("--bootstrap_samples", help="number of bootstrap samples.",
                        default=200, type=int) # What happens if I do 1000? samples, so far the analysis is with 200
    parser.add_argument("--bootstrap_method",
                        help="which bootstrap method to use. Options: 'quantile', 'BCa', 'se'",
                        default="quantile")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout for optimization in seconds")

    parser.add_argument("--model_comparisons", default="xgboost,lr",
                        help="For which models we want to get the significance test.")

    parser.add_argument("--alpha", default=5.0, type=float,
                        help="Alpha significance level.")
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
    set_seed(args.seed)

    main(args)



