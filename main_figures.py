# Main script for getting the main figures

import os
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
    'brown': '#D55E00',
    'magenta': '#CC79A7',
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
}

LABEL_ABBREVIATION_DICT = {
    "mental_stress": "MS",
    "baseline": "BASE",
    "high_physical_activity": "HPA",
    "moderate_physical_activity": "MPA",
    "low_physical_activity": "LPA",
}


def validate_models(models_str: str) -> list[str]:
    """Validate and convert comma-separated model string to list"""
    models = [model.strip().lower() for model in models_str.split(',')]
    valid_models = ['dt', 'rf', 'adaboost', 'lda', 'knn', 'lr', 'xgboost', 'qda', 'svm']

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
        save_name = f"None_{model_name}_bootstrapped.json" if not resampled else f"smote_{model_name}_bootstrapped.json"
    elif results_type == "feature_selection":
        folder_name = "feature_selection"
        save_name = "feature_selection_results.json"

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

    colors = {
        'rf': '#E69F00', #Orange
        'xgboost': '#56B4E9',  # Sky blue
        'lr': '#009E73', #Green
        'yellow': '#F0E442',
        'blue': '#0072B2', # Blue
        'brown': '#D55E00',
        'magenta': '#CC79A7',
    }

    # Get all sample frequencies and models (sorted)
    sample_freqs = sorted(bootstrapped_results.keys())
    all_models = list(set([model for freq_results in bootstrapped_results.values()
                          for model in freq_results.keys()]))

    # Calculate x-positions
    x = np.arange(len(sample_freqs))
    width = 0.3 / len(all_models)  # Reduced from 0.8 to 0.2 to bring bars closer together

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
                                color=colors[model], label=MODELS_ABBREVIATION_DICT[model],
                                elinewidth=2)

            # Add mean values very close to the points
            for i, (pos, mean) in enumerate(zip(x_pos[valid_idx], means[valid_idx])):
                plt.text(pos + width/36, mean, f' {mean:.3f}',  # Adjusted text position scaling
                        ha='left', va='center',
                        color='black',
                        fontsize=14,
                        weight="bold")

            handles.append(handle)

    # Customize plot
    plt.xlabel('Sampling Frequency (Hz)')

    # Simplified metric name on y-axis
    metric_labels = {
        'roc_auc': 'ROC-AUC',
        'pr_auc': 'PR-AUC',
        'precision': 'Precision',
        'balanced_accuracy': 'Balanced Accuracy'
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


def main(args):
    # Get all sample frequencies to analyze
    sample_frequencies = [128, 256, 512, 1000]  # Add or modify frequencies as needed

    comparison = f"{LABEL_ABBREVIATION_DICT[args.positive_class]}_{LABEL_ABBREVIATION_DICT[args.negative_class]}"
    figures_path = os.path.join(FIGURES_PATH, str(args.sample_frequency),
                                    str(args.window_size), comparison)
    # We use this to either get the results from smote or not
    resampled_bool = True if args.negative_class in ["low_physical_activity", "moderate_physical_activity"] else False

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

    plot_feature_selection(feature_selection_results, figures_path)
    # Plot bootstrap comparisons for each metric
    metrics = ['roc_auc', 'pr_auc', 'balanced_accuracy']
    for metric in metrics:
        plot_bootstrap_comparison(bootstrapped_results, metric, FIGURES_PATH, comparison)

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
    parser.add_argument("--window_size", type=int, default=60, help="The window size that we use for detecting stress")
    parser.add_argument('--window_shift', type=int, default=10,
                        help="The window shift that we use for detecting stress")
    parser.add_argument(
        "--models",
        help="Comma-separated list of models to analyze. Choose from: 'dt', 'rf', 'adaboost', 'lda', "
             "'knn', 'lr', 'xgboost', 'qda', 'svm'",
        type=validate_models,
        default="lr,xgboost,rf"
    )
    parser.add_argument("--bin_size", help="what bin size to use for plotting the calibration plots",
                        default=10, type=int)
    parser.add_argument("--bin_strategy", help="what binning strategy to use",
                        default="uniform", choices=("uniform", "quantile")
                        )
    
    args = parser.parse_args()
    main(args)

    #ToDo:
    # Plot feature selection history with #number of features against the feature
