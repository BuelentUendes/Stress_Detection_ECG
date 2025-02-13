# Main script for getting the main figures

import os
import argparse
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

def plot_combined_calibration_curves(models: list[str], n_bins: int, bin_strategy: str, figures_path: str) -> None:
    """
    Creates a combined calibration plot for multiple models.
    
    Args:
        models: List of model names to include
        n_bins: Number of bins for calibration curve
        bin_strategy: Strategy for binning ('uniform' or 'quantile')
        figures_path: Base path to figures directory
    """
    plt.figure(figsize=(10, 8))
    
    for model in models:
        # Construct path to calibration results
        model_cal_path = os.path.join(figures_path, model, f'{bin_strategy}_{n_bins}_calibration_summary.csv')
        
        try:
            # Load calibration data
            cal_df = pd.read_csv(model_cal_path)
            ece = cal_df["ece"][0]

            # Plot calibration curve
            plt.plot(
                cal_df['prob_pred'], 
                cal_df['prob_true'], 
                marker='o', 
                linewidth=1, 
                label=f"{MODELS_ABBREVIATION_DICT[model]} ECE: {np.round(ece, 4)}"
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
    save_path = os.path.join(figures_path, f'combined_calibration_curves_{bin_strategy}_{n_bins}.png')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()


def main(args):
    # Create combined calibration plots
    # Create path folder depending on the comparison we are trying to do
    comparison = f"{LABEL_ABBREVIATION_DICT[args.positive_class]}_{LABEL_ABBREVIATION_DICT[args.negative_class]}"
    figures_path_root = os.path.join(FIGURES_PATH, str(args.sample_frequency), str(args.window_size), comparison)

    plot_combined_calibration_curves(
        models=args.models,
        n_bins=args.bin_size,
        bin_strategy=args.bin_strategy,
        figures_path=figures_path_root
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
        default="lr,rf,xgboost"
    )
    parser.add_argument("--bin_size", help="what bin size to use for plotting the calibration plots",
                        default=10, type=int)
    parser.add_argument("--bin_strategy", help="what binning strategy to use",
                        default="uniform", choices=("uniform", "quantile")
                        )
    
    args = parser.parse_args()
    main(args)