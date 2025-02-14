"""
Main script for generating publication-quality figures from machine learning results.
"""

import os
import argparse
import json
from typing import Optional, Tuple, Union, Any, Dict, List
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils.helper_path import CLEANED_DATA_PATH, FEATURE_DATA_PATH, MODELS_PATH, CONFIG_PATH, RESULTS_PATH, FIGURES_PATH
from utils.helper_argparse import (
    validate_scaler, validate_category, validate_target_metric, 
    validate_ml_model, validate_resampling_method
)

# Type aliases
ResultsDict = Dict[str, Dict[str, Union[float, Dict[str, float]]]]
ColorDict = Dict[str, str]

# Constants
COLORS: ColorDict = {
    'rf': '#E69F00',  # Orange
    'xgboost': '#56B4E9',  # Sky blue
    'lr': '#009E73',  # Green
    'yellow': '#F0E442',
    'blue': '#0072B2',  # Blue
    'dt': '#D55E00',
    'adaboost': '#CC79A7',
}

MODELS_ABBREVIATION_DICT: Dict[str, str] = {
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

LABEL_ABBREVIATION_DICT: Dict[str, str] = {
    "mental_stress": "MS",
    "baseline": "BASE",
    "high_physical_activity": "HPA",
    "moderate_physical_activity": "MPA",
    "low_physical_activity": "LPA",
}

METRIC_LABELS: Dict[str, str] = {
    'roc_auc': 'ROC-AUC',
    'pr_auc': 'PR-AUC',
    'precision': 'Precision',
    'balanced_accuracy': 'Balanced Accuracy'
}


def set_plot_style() -> None:
    """Set the matplotlib style for publication-quality figures."""
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


def validate_models(models_str: str) -> List[str]:
    """Validate and convert comma-separated model string to list."""
    models = [model.strip().lower() for model in models_str.split(',')]
    valid_models = ['dt', 'rf', 'adaboost', 'lda', 'knn', 'lr', 'xgboost', 'qda', 'svm']
    
    for model in models:
        if model not in valid_models:
            raise argparse.ArgumentTypeError(
                f"Invalid model: {model}. Choose from: {', '.join(valid_models)}"
            )
    return models


def load_bootstrap_results(
    path: str, 
    model_name: str, 
    sample_freq: int, 
    window_size: int, 
    comparison: str
) -> Optional[Dict]:
    """Load bootstrap results for a specific model and sample frequency."""
    full_path = os.path.join(
        path, str(sample_freq), str(window_size), comparison,
        model_name, "bootstrap_test", f"None_{model_name}_bootstrapped.json"
    )
    try:
        with open(full_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results for {model_name} at {sample_freq}Hz: {e}")
        return None


def plot_bootstrap_comparison(
    bootstrapped_results: Dict[int, Dict[str, Dict]], 
    metric: str, 
    figures_path_root: str
) -> None:
    """
    Plot bootstrap results comparison across sample frequencies for multiple models.
    
    Args:
        bootstrapped_results: Nested dict {sample_freq: {model: results}}
        metric: Metric to plot ('roc_auc', 'pr_auc', 'precision')
        figures_path_root: Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    set_plot_style()
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Get all sample frequencies and models (sorted)
    sample_freqs = sorted(bootstrapped_results.keys())
    all_models = list(set([model for freq_results in bootstrapped_results.values() 
                          for model in freq_results.keys()]))
    
    # Plot settings
    x = np.arange(len(sample_freqs))
    width = 0.2 / len(all_models)
    
    handles = plot_model_results(bootstrapped_results, all_models, sample_freqs, x, width, metric)
    
    customize_plot(x, sample_freqs, metric)
    save_plot(figures_path_root, metric)


def plot_model_results(
    bootstrapped_results: Dict, 
    all_models: List[str], 
    sample_freqs: List[int], 
    x: np.ndarray, 
    width: float, 
    metric: str
) -> List:
    """Plot results for each model."""
    handles = []
    for idx, model in enumerate(all_models):
        means, ci_lower, ci_upper = collect_model_data(bootstrapped_results, model, sample_freqs, metric)
        
        x_pos = x + (idx - len(all_models)/2 + 0.5) * width
        valid_idx = ~np.isnan(means)
        
        if np.any(valid_idx):
            handle = plot_single_model(x_pos, means, ci_lower, ci_upper, valid_idx, model, width)
            handles.append(handle)
    
    return handles


def collect_model_data(
    bootstrapped_results: Dict, 
    model: str, 
    sample_freqs: List[int], 
    metric: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect statistical data for a specific model."""
    means, ci_lower, ci_upper = [], [], []
    
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
    
    return map(np.array, (means, ci_lower, ci_upper))


def plot_single_model(
    x_pos: np.ndarray, 
    means: np.ndarray, 
    ci_lower: np.ndarray, 
    ci_upper: np.ndarray, 
    valid_idx: np.ndarray, 
    model: str, 
    width: float
):
    """Plot data for a single model."""
    handle = plt.errorbar(
        x_pos[valid_idx], means[valid_idx],
        yerr=[means[valid_idx] - ci_lower[valid_idx],
              ci_upper[valid_idx] - means[valid_idx]],
        fmt='o', capsize=5, capthick=2, markersize=8,
        color=COLORS[model], label=MODELS_ABBREVIATION_DICT[model],
        elinewidth=2
    )
    
    # Add mean values
    for pos, mean in zip(x_pos[valid_idx], means[valid_idx]):
        plt.text(pos + width/36, mean, f' {mean:.3f}',
                ha='left', va='center',
                color='black',
                weight="bold",
                fontsize=14)
    
    return handle


def customize_plot(x: np.ndarray, sample_freqs: List[int], metric: str) -> None:
    """Customize plot appearance."""
    plt.xlabel('Sampling Frequency (Hz)')
    plt.ylabel(METRIC_LABELS.get(metric, metric))
    plt.xticks(x, [str(freq) for freq in sample_freqs])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()


def save_plot(figures_path_root: str, metric: str) -> None:
    """Save the plot to file."""
    save_path = os.path.join(figures_path_root, f'bootstrap_comparison_{metric}_multi_freq.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
    plt.close()


def main(args: argparse.Namespace) -> None:
    """Main execution function."""
    sample_frequencies = [128, 256, 512, 1000]
    comparison = f"{LABEL_ABBREVIATION_DICT[args.positive_class]}_{LABEL_ABBREVIATION_DICT[args.negative_class]}"
    
    # Collect results for all frequencies
    bootstrapped_results = {
        freq: {
            model: load_bootstrap_results(
                RESULTS_PATH, model, freq, args.window_size, comparison
            ) for model in args.models
        } for freq in sample_frequencies
    }
    
    # Generate plots
    for metric in ['roc_auc', 'pr_auc', 'balanced_accuracy']:
        plot_bootstrap_comparison(bootstrapped_results, metric, FIGURES_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication-quality figures from ML results.")
    parser.add_argument("--seed", help="seed number", default=42, type=int)
    parser.add_argument("--positive_class", help="Which category should be 1", 
                       default="mental_stress", type=validate_category)
    parser.add_argument("--negative_class", help="Which category should be 0", 
                       default="baseline", type=validate_category)
    parser.add_argument("--sample_frequency", help="which sample frequency to use for the training",
                       default=1000, type=int)
    parser.add_argument("--window_size", type=int, default=60, 
                       help="The window size that we use for detecting stress")
    parser.add_argument('--window_shift', type=int, default=10,
                       help="The window shift that we use for detecting stress")
    parser.add_argument("--models", help="Comma-separated list of models to analyze",
                       type=validate_models, default="lr,xgboost,rf")
    parser.add_argument("--bin_size", help="what bin size to use for plotting the calibration plots",
                       default=10, type=int)
    parser.add_argument("--bin_strategy", help="what binning strategy to use",
                       default="uniform", choices=("uniform", "quantile"))
    
    args = parser.parse_args()
    main(args)