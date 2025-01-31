# Simple script to train machine learning models on the stress dataset

import os
import argparse

import torch
import numpy as np
import random

from utils.helper_path import CLEANED_DATA_PATH, FEATURE_DATA_PATH
from utils.helper_functions import set_seed, get_data_folders, ECGDataset, encode_data, prepare_data


def validate_scaler(value:str) -> str:
    if value not in ["standard_scaler", "min_max", None]:
        raise argparse.ArgumentTypeError(f"Invalid choice: {value}. Choose from 'standard_scaler' or 'min_max'.")
    return value


def validate_category(value:str) -> str:
    valid_categories = ['high_physical_activity', 'mental_stress', 'baseline',
                         'low_physical_activity', 'moderate_physical_activity']
    if value not in valid_categories:
        raise argparse.ArgumentTypeError(f"Invalid choice: {value}. "
                                         f"Choose from options in {valid_categories}.")
    return value


def main(args):
    target_data_path = os.path.join(FEATURE_DATA_PATH, str(args.sample_frequency))
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed number", default=42, type=int)
    parser.add_argument("--positive_class", help="Which category should be 1", default="mental_stress",
                        type=validate_category)
    parser.add_argument("--negative_class", help="Which category should be 0", default="baseline",
                        type=validate_category)
    parser.add_argument("--standard_scaler", help="Which standard scaler to use", type=validate_scaler,
                        default=None)
    parser.add_argument("--sample_frequency", help="which sample frequency to use for the training",
                        default=1000, type=int)
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    main(args)


