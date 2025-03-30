import json
from tqdm import tqdm
import os
import pandas as pd
import argparse
from datetime import timedelta
import sys
from helper_path import CLEANED_DATA_PATH


def calculate_sampling_frequency(df, timestamp_column):
    """
    Calculate the actual sampling frequency from the timestamp column.

    Args:
        df (pandas.DataFrame): DataFrame containing the time series data
        timestamp_column (str): Name of the column containing timestamps

    Returns:
        float: Calculated sampling frequency in Hz
    """
    # Sort by timestamp to ensure correct calculation
    df = df.sort_values(by=timestamp_column)

    # Calculate time differences between consecutive samples in seconds
    time_diffs = df[timestamp_column].diff().dropna()

    if time_diffs.empty:
        raise ValueError("Cannot calculate frequency: Not enough timestamps")

    # Calculate the median time difference to avoid outliers
    median_diff_seconds = time_diffs.median().total_seconds()

    if median_diff_seconds == 0:
        raise ValueError("Cannot calculate frequency: Zero time difference between samples")

    # Calculate frequency (samples per second = Hz)
    actual_frequency = 1 / median_diff_seconds

    return actual_frequency


def verify_sampling_frequency(expected_frequency, timestamp_column, tolerance=0.1,
                              participant_id_filename=30100,
                              root_path=CLEANED_DATA_PATH,
                              verbose=False):
    """
    Verify if the actual sampling frequency matches the expected one.

    Args:
        expected_frequency (int): Expected sampling frequency in Hz
        timestamp_column (str): Name of the column containing timestamps
        tolerance (float): Acceptable deviation from expected frequency (proportion)
        participant_id_filename (str): file name, should in form <ID>.parquet
        root_path (str): Root path for cleaned dataset
        verbose (bool): If we want to have a verbose output

    Returns:
        bool: True if the frequency matches within tolerance, False otherwise
    """
    try:
        # Load the parquet file
        df = pd.read_parquet(os.path.join(root_path, str(expected_frequency), participant_id_filename))

        # Check if timestamp column exists
        if timestamp_column not in df.columns:
            print(f"Error: Timestamp column '{timestamp_column}' not found in the file.")
            print(f"Available columns: {', '.join(df.columns)}")
            return False

        # Ensure timestamp column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            print(f"Converting '{timestamp_column}' to datetime format...")
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        # Calculate actual frequency
        actual_frequency = calculate_sampling_frequency(df, timestamp_column)

        # Check if frequency matches within tolerance
        lower_bound = expected_frequency * (1 - tolerance)
        upper_bound = expected_frequency * (1 + tolerance)
        matches = lower_bound <= actual_frequency <= upper_bound

        # Print results
        if verbose:
            print(f"Participant id: {participant_id_filename.split('.')[0]}")
            print(f"Expected frequency: {expected_frequency:.2f} Hz")
            print(f"Actual frequency: {actual_frequency:.2f} Hz")
            print(f"Tolerance range: {lower_bound:.2f} - {upper_bound:.2f} Hz")
            print(f"Result: {'MATCH' if matches else 'MISMATCH'}")

        return actual_frequency, matches

    except Exception as e:
        print(f"Error processing file {participant_id_filename.split('.')[0]}: {str(e)}")
        return None, False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Verify sampling frequency of parquet files.')
    parser.add_argument('--expected_frequency', type=int, help='Expected sampling frequency in Hz',
                        default=1_000)
    parser.add_argument('--timestamp_column', default='timestamp',
                        help='Name of the timestamp column (default: timestamp)')
    parser.add_argument('--tolerance', type=float, default=0.05,
                        help='Acceptable tolerance as a proportion (default: 0.1)')
    parser.add_argument('--participant_id', type=int,
                        help="Which id to check. "
                             "If set to -1 we go through all participants and store the results in a log file.",
                        default=-1)

    args = parser.parse_args()

    if args.participant_id == -1:
        folderpath = os.path.join(CLEANED_DATA_PATH, str(args.expected_frequency))
        files = [f for f in os.listdir(folderpath) if f.endswith('.parquet')]

        participant_downsampling_results = {}
        for file in tqdm(files, desc="Processing participants", unit="file"):
            participant_id = file.split(".")[0]
            actual_frequency, match_result = verify_sampling_frequency(
                args.expected_frequency,
                args.timestamp_column,
                args.tolerance,
                participant_id_filename=file,
            )

            participant_downsampling_results[participant_id] = {
                'actual_frequency': actual_frequency,
                'match_result': match_result
            }

        with open(os.path.join(folderpath, "check_downsampling_results.json"), "w") as f:
            json.dump(participant_downsampling_results, f, indent=4)

    else:
        # Verify the sampling frequency
        actual_frequency, match_result = verify_sampling_frequency(
            args.expected_frequency,
            args.timestamp_column,
            args.tolerance,
            participant_id_filename=f"{str(args.participant_id)}.parquet",
        )

    # Exit with appropriate code
    sys.exit(0 if match_result else 1)

if __name__ == "__main__":
    main()