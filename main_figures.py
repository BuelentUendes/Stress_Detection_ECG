# Main script for getting the main figures

import os
import re
import argparse
import json
from typing import Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings("ignore")

import torch

from utils.helper_path import CLEANED_DATA_PATH, FEATURE_DATA_PATH, MODELS_PATH, CONFIG_PATH, RESULTS_PATH, FIGURES_PATH
from utils.helper_argparse import validate_scaler, validate_category, validate_target_metric, validate_ml_model, \
    validate_resampling_method

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Color scheme:
COLORS_DICT = {
    'rf': '#E69F00',  # Orange
    'xgboost': '#56B4E9',  # Sky blue
    'lr': '#009E73',  # Green
    'yellow': '#F0E442',
    'blue': '#0072B2',  # Blue
    'random_baseline': '#D55E00',
    'gmm': '#CC79A7',
    "simple_baseline": '#CC79A7',
}

MODELS_ABBREVIATION_DICT = {
    "lr": "Logistic Regression",
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
    "simple_baseline": "Simple LR baseline"
}

LABEL_ABBREVIATION_DICT = {
    "mental_stress": "MS",
    "baseline": "BASE",
    "high_physical_activity": "HPA",
    "moderate_physical_activity": "MPA",
    "low_physical_activity": "LPA",
    "rest": "REST",
    "any_physical_activity": "ANY_PHY",
}


def validate_models(models_str: str) -> list[str]:
    """Validate and convert comma-separated model string to list"""
    models = [model.strip().lower() for model in models_str.split(',')]
    valid_models = ['dt', 'rf', 'adaboost', 'lda', 'knn', 'lr', 'xgboost', 'qda', 'svm', 'random_baseline',
                    'gmm', 'simple_baseline']

    for model in models:
        if model not in valid_models:
            raise argparse.ArgumentTypeError(
                f"Invalid model: {model}. Choose from: {', '.join(valid_models)}"
            )
    return models


def plot_combined_calibration_curves(models: list[str], n_bins: int, bin_strategy: str,
                                     figures_path: str, comparison: str) -> None:
    """
    Creates a combined calibration plot for multiple models.

    Args:
        models: List of model names to include
        n_bins: Number of bins for calibration curve
        bin_strategy: Strategy for binning ('uniform' or 'quantile')
        figures_path: Base path to figures directory
        comparison: What comparison is plotted
    """
    plt.figure(figsize=(10, 8))

    for model in models:
        # Construct path to calibration results
        model_cal_path = os.path.join(figures_path, model, f'{bin_strategy}_{n_bins}_calibration_summary.csv')

        try:
            # Load calibration data
            cal_df = pd.read_csv(model_cal_path)
            ece = cal_df["ece"][0]
            brier_score = cal_df["brier score"][0]

            # Plot calibration curve
            plt.plot(
                cal_df['prob_pred'],
                cal_df['prob_true'],
                marker='o',
                linewidth=1,
                color=COLORS_DICT[model],
                label=f"{MODELS_ABBREVIATION_DICT[model]} ECE: {np.round(ece, 4)} "
                      f"Brier score: {np.round(brier_score, 4)}"
            )
        except Exception as e:
            print(f"Error loading calibration data for {model}: {e}. We will continue")
            continue

    # Add diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability in Each Bin')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(loc='upper left')
    plt.tight_layout()

    # Save the combined plot
    save_path = os.path.join(figures_path, f'{comparison}_combined_calibration_curves_{bin_strategy}_{n_bins}.png')
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()


def load_json_feature_selection_results(
        path: str,
        model_name: str,
        sample_freq: int,
        window_size: int,
        comparison: str,
        resampled: Optional[bool] = False
):
    root_path = os.path.join(path, str(sample_freq), str(window_size), comparison,
                            model_name)

    bootstrap_path = os.path.join(root_path, "bootstrap_test")
    feature_selection_path = os.path.join(root_path, "feature_selection")
    #Get first the feature selection results to extract the number of features used:
    try:
        with open(os.path.join(feature_selection_path, "feature_selection_results.json"), 'r') as f:
            feature_selection_results = json.load(f)
            best_number_features = feature_selection_results["best_n_features"]
    except FileNotFoundError:
        best_number_features = None
        print(f"We could not find the file. We will continue.")

    # Now load the best performance together with the feature selection results
    prefix = "None" if not resampled else "smote"
    middle_suffix = ["bootstrapped", "feature_selection_bootstrapped", "subset_features_bootstrapped",
                     "subset_features_random_bootstrapped"]

    middle_suffix = [
        "bootstrapped",
        "feature_selection_subset_features_5_bootstrapped",
        "feature_selection_subset_features_10_bootstrapped",
        "feature_selection_subset_features_20_bootstrapped"
    ]

    bootstrap_results = {}
    for suffix in middle_suffix:
        save_name = f"{prefix}_{model_name}_{suffix}.json"
        feature_number_dict = {
            "bootstrapped": 104,
            "feature_selection_subset_features_20_bootstrapped": 20,
            "feature_selection_subset_features_10_bootstrapped": 10,
            "feature_selection_subset_features_5_bootstrapped": 5,
        }
        feature_number = feature_number_dict[suffix]
        try:
            with open(os.path.join(bootstrap_path, save_name), 'r') as f:
                if not "random" in suffix:
                    bootstrap_results[f"feature number {feature_number}"] = json.load(f)
                else:
                    bootstrap_results[f"feature number random {feature_number}"] = json.load(f)
        except FileNotFoundError:
            print(f"We could not find the file. We will continue.")

    return bootstrap_results


def load_json_results(path: str,
                      model_name: str,
                      sample_freq: int,
                      window_size: int,
                      comparison: str,
                      results_type: str,
                      resampled: Optional[bool] = False,) -> dict:
    """Load bootstrap results or feature selection results for a specific model and sample frequency"""

    if results_type == "bootstrap":
        folder_name = "bootstrap_test"
        # We only ran the gaussian mixture model with 2 components for one feature subset
        if model_name == "gmm":
            save_name = f"None_{model_name}_subset_features_bootstrapped.json" if not resampled \
                else f"smote_{model_name}_subset_features_bootstrapped.json"

        elif model_name == "simple_baseline":
            save_name = f"None_lr_subset_features_bootstrapped.json" if not resampled \
                else f"smote_lr_subset_features_bootstrapped.json"

        else:
            save_name = f"None_{model_name}_bootstrapped.json" if not resampled else f"smote_{model_name}_bootstrapped.json"
    elif results_type == "feature_selection":
        folder_name = "feature_selection"
        save_name = "feature_selection_results.json"

    if model_name == "simple_baseline":
        model_name = "lr"

    full_path = os.path.join(path, str(sample_freq), str(window_size), comparison,
                            model_name, f"{folder_name}", f"{save_name}")

    try:
        with open(full_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results for {model_name} at {sample_freq}Hz: {e}")
        return None


def plot_bootstrap_comparison(bootstrapped_results: dict, metric: str, figures_path_root: str, comparison: str) -> None:
    """
    Plot bootstrap results comparison across sample frequencies for multiple models.

    Args:
        bootstrapped_results: Nested dict {sample_freq: {model: results}}
        metric: Metric to plot ('roc_auc', 'pr_auc', 'precision')
        figures_path_root: Path to save the figure
        comparison: What comparison is plotted
    """
    plt.figure(figsize=(12, 8))

    # Set figure style for publication
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'Arial',
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'figure.dpi': 300,
    })

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Get all sample frequencies and models (sorted)
    sample_freqs = sorted(bootstrapped_results.keys())
    all_models = list(set([model for freq_results in bootstrapped_results.values()
                          for model in freq_results.keys()]))

    # Calculate x-positions
    x = np.arange(len(sample_freqs))
    width = 0.3 / len(all_models)  # Adjusted bar width for better spacing

    # Plot for each model
    handles = []
    for idx, model in enumerate(all_models):
        means = []
        ci_lower = []
        ci_upper = []

        # Collect data for this model across all frequencies
        for freq in sample_freqs:
            results = bootstrapped_results[freq].get(model)
            if results and metric in results:
                means.append(results[metric]['mean'])
                ci_lower.append(results[metric]['ci_lower'])
                ci_upper.append(results[metric]['ci_upper'])
            else:
                means.append(np.nan)
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)

        # Convert to numpy arrays
        means = np.array(means)
        ci_lower = np.array(ci_lower)
        ci_upper = np.array(ci_upper)

        # Calculate x positions for this model (centered around the frequency position)
        x_pos = x + (idx - len(all_models)/2 + 0.5) * width

        # Plot confidence intervals and means
        valid_idx = ~np.isnan(means)
        if np.any(valid_idx):
            handle = plt.errorbar(x_pos[valid_idx], means[valid_idx],
                                yerr=[means[valid_idx] - ci_lower[valid_idx],
                                     ci_upper[valid_idx] - means[valid_idx]],
                                fmt='o', capsize=5, capthick=2, markersize=8,
                                color=COLORS_DICT[model], label=MODELS_ABBREVIATION_DICT[model],
                                elinewidth=2)

            # Adjusted label placement to avoid overlapping
            y_positions = []  # Track annotated y-positions
            offset = 0.175 * (max(means) - min(means))  # Adaptive offset for better spacing

            for i, (pos, mean) in enumerate(zip(x_pos[valid_idx], means[valid_idx])):
                new_y = mean
                while any(abs(new_y - y) < offset for y in y_positions):
                    new_y -= offset  # Shift up if overlap detected

                plt.text(pos + width/36, new_y, f' {mean:.3f}',
                         ha='left', va='center',
                         color='black',
                         fontsize=14,
                         weight="bold")

                y_positions.append(new_y)  # Store adjusted y-position

            handles.append(handle)

    # Customize plot
    plt.xlabel('Sampling Frequency (Hz)')

    # Simplified metric name on y-axis
    metric_labels = {
        'roc_auc': 'ROC-AUC',
        'pr_auc': 'PR-AUC',
        'precision': 'Precision',
        'balanced_accuracy': 'Balanced Accuracy',
        'f1_score': "F1-Score"
    }
    plt.ylabel(metric_labels.get(metric, metric))

    # Set x-ticks to sample frequencies
    plt.xticks(x, [str(freq) for freq in sample_freqs])

    # Add legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.5)

    # Adjust layout and save
    plt.tight_layout()
    save_path = os.path.join(figures_path_root, f'{comparison}_bootstrap_comparison_{metric}_multi_freq.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
    plt.close()


def plot_feature_selection(
        feature_selection_dict: dict,
        figures_path_root: str,
        start_feature_selection: int = 5
) -> None:

    x_axis = np.arange(start=start_feature_selection,
                       stop=start_feature_selection + len(next(iter(feature_selection_dict.values()))))

    plt.figure(figsize=(8, 6))

    for model, scores in feature_selection_dict.items():
        # Plot calibration curve
        plt.plot(x_axis, scores,
            color=COLORS_DICT[model],
            label=f"{MODELS_ABBREVIATION_DICT[model]}"
        )

    plt.xlabel('Number of features')
    plt.ylabel('Validation ROC-AUC score')

    plt.legend(loc='lower right')
    plt.tight_layout()

    # Save the combined plot
    save_path = os.path.join(figures_path_root, f'feature_selection.png')
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()


def plot_feature_subset_comparison(results: dict, metric: str, figures_path_root: str, comparison: str) -> None:
    """
    Plot model performance across feature subsets with confidence intervals.

    Args:
        results: Nested dict {model_type: {feature_set: results_dict}}
        metric: Performance metric to plot ('roc_auc', 'pr_auc', etc.)
        figures_path_root: Path to save the figure.
        comparison: What comparison is plotted.
    """
    plt.figure(figsize=(12, 8))

    # Set figure style for publication
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'Arial',
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'figure.dpi': 300,
    })

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Get unique models and feature sets
    model_types = sorted(results.keys())
    feature_sets = sorted({fs for model in results.values() for fs in model.keys()})
    sorted_features = sorted(feature_sets, key=lambda x: int(re.search(r'\d+', x).group()), reverse=True)

    # Reduce spacing between models
    x = np.arange(len(model_types)) * 0.8  # Reduce model spacing
    width = 0.08  # Smaller width for tighter grouping

    handles = []
    for idx, feature_set in enumerate(sorted_features):
        means, ci_lower, ci_upper = [], [], []

        for model_type in model_types:
            result = results[model_type].get(feature_set)
            if result and metric in result:
                means.append(result[metric]['mean'])
                ci_lower.append(result[metric]['ci_lower'])
                ci_upper.append(result[metric]['ci_upper'])
            else:
                means.append(np.nan)
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)

        means = np.array(means)
        ci_lower = np.array(ci_lower)
        ci_upper = np.array(ci_upper)

        # Compute x positions
        x_pos = x + (idx - len(feature_sets)/2 + 0.5) * width

        valid_idx = ~np.isnan(means)
        if np.any(valid_idx):
            handle = plt.errorbar(x_pos[valid_idx], means[valid_idx],
                                  yerr=[means[valid_idx] - ci_lower[valid_idx],
                                        ci_upper[valid_idx] - means[valid_idx]],
                                  fmt='o', capsize=4, capthick=1.8, markersize=7,
                                  label=feature_set, elinewidth=1.8)

            # Prevent overlapping labels
            y_positions = []
            offset = 0.015 * (max(means[valid_idx]) - min(means[valid_idx]))

            for pos, mean in zip(x_pos[valid_idx], means[valid_idx]):
                new_y = mean
                while any(abs(new_y - y) < offset for y in y_positions):
                    new_y -= offset

                plt.text(pos, new_y, f'{mean:.3f}', ha='center', va='bottom',
                         color='black', fontsize=12, fontweight="bold")
                y_positions.append(new_y)

            handles.append(handle)

    # Customize plot
    plt.xlabel('Model Type')
    plt.ylabel(metric.upper().replace('_', '-'))

    # Set x-ticks to model types
    plt.xticks(x, model_types)

    # Adjust legend positioning
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title="Number of Features")

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save figure
    plt.tight_layout()
    save_path = os.path.join(figures_path_root, f'{comparison}_feature_subset_comparison_{metric}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
    plt.close()


def main(args):
    # Get all sample frequencies to analyze
    sample_frequencies = [128, 256, 512, 1000]  # Add or modify frequencies as needed

    comparison = f"{LABEL_ABBREVIATION_DICT[args.positive_class]}_{LABEL_ABBREVIATION_DICT[args.negative_class]}"
    figures_path = os.path.join(FIGURES_PATH, str(args.sample_frequency),
                                    str(args.window_size), comparison)
    # We use this to either get the results from smote or not
    # Actually, smote does not really have an impact on the performance.

    resampled_bool = (args.negative_class in [
        "low_physical_activity",
        "moderate_physical_activity",
        "rest",
        "any_physical_activity"
    ]) or (args.positive_class in [
        "mental_stress",
        "low_physical_activity",
        "moderate_physical_activity",
        "rest",
        "any_physical_activity"
    ])

    # Collect results for all frequencies
    bootstrapped_results = {}
    feature_selection_results = {}

    for freq in sample_frequencies:
        bootstrapped_results[freq] = {
            model: load_json_results(
                RESULTS_PATH,
                model,
                freq,
                args.window_size,
                comparison,
                "bootstrap",
                resampled_bool,
            )
            for model in args.models
        }

    for model in args.models:
        try:
            feature_selection_results[model] = load_json_results(
                    RESULTS_PATH,
                    model,
                    sample_freq=1000,
                    window_size=args.window_size,
                    comparison=comparison,
                    results_type="feature_selection",
                )["scores"]
        except TypeError:
            print(f"{model} does not have the results, we skip it.")
            continue

    try:
        plot_feature_selection(feature_selection_results, figures_path)
    except StopIteration:
        print(f"We could not find the file")

    performance_results_overview = {
        model: load_json_feature_selection_results(
            RESULTS_PATH, model, 1000, args.window_size, comparison
        ) for model in args.models
    }

    # Plot bootstrap comparisons for each metric
    metrics = ['roc_auc', 'pr_auc', 'balanced_accuracy', 'f1_score']
    for metric in metrics:
        plot_bootstrap_comparison(bootstrapped_results, metric, FIGURES_PATH, comparison)
        plot_feature_subset_comparison(
            results=performance_results_overview,
            metric=metric,
            figures_path_root=figures_path,
            comparison='model_feature_comparison'
        )

    # Note: The calibration curves plotting remains unchanged as it's for single frequency
    if args.sample_frequency in sample_frequencies:
        plot_combined_calibration_curves(
            models=args.models,
            n_bins=args.bin_size,
            bin_strategy=args.bin_strategy,
            figures_path=figures_path,
            comparison=comparison,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed number", default=42, type=int)
    parser.add_argument("--positive_class", help="Which category should be 1", default="mental_stress",
                        type=validate_category)
    parser.add_argument("--negative_class", help="Which category should be 0", default="baseline",
                        type=validate_category)
    parser.add_argument("--sample_frequency", help="which sample frequency to use for the training",
                        default=1000, type=int)
    parser.add_argument("--window_size", type=int, default=30, help="The window size that we use for detecting stress")
    parser.add_argument('--window_shift', type=int, default=10,
                        help="The window shift that we use for detecting stress")
    parser.add_argument(
        "--models",
        help="Comma-separated list of models to analyze. Choose from: 'dt', 'rf', 'adaboost', 'lda', "
             "'knn', 'lr', 'xgboost', 'qda', 'svm', 'random_baseline', 'gmm', 'simple_baseline'",
        type=validate_models,
        default="lr, rf, xgboost"
    )
    parser.add_argument("--bin_size", help="what bin size to use for plotting the calibration plots",
                        default=10, type=int)
    parser.add_argument("--bin_strategy", help="what binning strategy to use",
                        default="uniform", choices=("uniform", "quantile")
                        )

    args = parser.parse_args()
    main(args)

    #ToDo:
    # Here we assume that we use SMOTE anyways for mental stress vs baseline, yet, I think there are no
    # performance differences between the two
    # However, the calibration changes accordinlgy (less-well calibrated when I use SMOTE compared to non-oversampling)

