import os
from pathlib import Path
from typing import List, Dict, Any
import argparse

import neurokit2 as nk
import numpy as np
from pyedflib import highlevel
from tqdm import tqdm

from utils.helper_path import RAW_DATA_PATH, CLEANED_DATA_PATH

def find_all_edf_files(directory: str) -> List[str]:
    """
    Find all EDF files in the given directory.
    
    Args:
        directory: Path to the directory to search for EDF files
        
    Returns:
        List of full paths to EDF files
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.edf')]


def create_directory(path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def downsample_ecg_file(
    input_path: str, 
    output_path: str, 
    desired_sampling_rate: int
) -> None:
    """
    Downsample an ECG signal from an EDF file and save it.
    
    Args:
        input_path: Path to input EDF file
        output_path: Path where downsampled EDF file should be saved
        desired_sampling_rate: Target sampling rate in Hz
        
    Notes:
        - Assumes ECG signal is the first channel in the EDF file
        - Original sampling rate is assumed to be 1000 Hz
        - NaN values are replaced with zeros
    """
    signals, signal_headers, header = highlevel.read_edf(input_path)
    
    # Clean and downsample the ECG signal
    ecg_signal = nk.signal_sanitize(signals[0])
    ecg_signal = np.nan_to_num(ecg_signal)
    cleaned_signal = nk.ecg_clean(ecg_signal, sampling_rate=1000)
    cleaned_signal = np.nan_to_num(cleaned_signal)
    downsampled_ecg = nk.signal_resample(
        cleaned_signal, 
        sampling_rate=1000,
        desired_sampling_rate=desired_sampling_rate
    )
    downsampled_ecg = np.nan_to_num(downsampled_ecg)

    # Update header for the new sampling rate
    new_header = signal_headers[0].copy()
    new_header['sample_rate'] = desired_sampling_rate
    new_header["sample_frequency"] = desired_sampling_rate

    # Write the downsampled EDF file
    highlevel.write_edf(output_path, np.array([downsampled_ecg]), [new_header], header)


def main(args: argparse.Namespace) -> None:
    """
    Main function to process all EDF files in the input directory.
    
    Args:
        args: Command line arguments containing desired_sampling_rate
    """
    edf_files = find_all_edf_files(RAW_DATA_PATH)
    output_path = os.path.join(CLEANED_DATA_PATH, str(args.desired_sampling_rate))
    create_directory(output_path)

    for edf_file in tqdm(edf_files, desc="Processing EDF files"):
        save_path = os.path.join(output_path, os.path.basename(edf_file))
        downsample_ecg_file(edf_file, save_path, args.desired_sampling_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Simple script to downsample ECG signals and save them in EDF files "
            "for which we can then use the preprocessing pipeline."
        )
    )
    parser.add_argument(
        "--desired_sampling_rate",
        type=int,
        help="Desired sampling rate for downsampling in Hz.",
        default=100
    )

    args = parser.parse_args()
    main(args)