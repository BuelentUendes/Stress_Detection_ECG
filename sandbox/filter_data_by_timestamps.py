import argparse
import os

from tqdm import tqdm
import pandas as pd
from utils.helper_path import  CLEANED_DATA_PATH

def filter_data_by_timestamps(data_df: pd.DataFrame, timestamps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the rows in data_df so that only those falling within the corresponding experimental
    time intervals are kept. The matching is based on the condition label.

    Parameters:
        data_df (pd.DataFrame): DataFrame with ECG data. Expected to contain a datetime column 'timestamp'
                                and a condition column 'label'.
        timestamps_df (pd.DataFrame): DataFrame with condition intervals. Expected to contain:
                                      - 'Category': the experimental condition name,
                                      - 'LabelStart': the start datetime of the condition,
                                      - 'LabelEnd': the end datetime of the condition.

    Returns:
        pd.DataFrame: A new DataFrame containing only rows from data_df whose 'timestamp' falls between
                      the corresponding 'LabelStart' and 'LabelEnd' for that condition.
    """
    # Force conversion of 'timestamp' to a timezone aware datetime64[ns, UTC]
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'].astype(str), utc=True)

    # # Ensure that the 'timestamp' column is in datetime format.
    # if not pd.api.types.is_datetime64_any_dtype(data_df['timestamp']):
    #     data_df['timestamp'] = pd.to_datetime(data_df['timestamp'], utc=True)

    # List to store filtered data segments for each condition interval.
    filtered_segments = []

    # Iterate over each row in the timestamps DataFrame.
    for _, row in timestamps_df.iterrows():
        condition = row['Category']
        start = row['LabelStart']
        end = row['LabelEnd']

        # Create a mask: match the condition and ensure the timestamp is within the interval.
        mask = (
                (data_df['label'] == condition) &
                (data_df['timestamp'] >= start) &
                (data_df['timestamp'] <= end)
        )
        segment = data_df[mask]

        filtered_segments.append(segment)

    # Concatenate all segments into a single DataFrame.
    if filtered_segments:
        filtered_data = pd.concat(filtered_segments, ignore_index=True)
    else:
        # If no segments were found, return an empty DataFrame with the same columns as data_df.
        filtered_data = data_df.iloc[0:0]

    return filtered_data


# Example usage:
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Preprocessing pipeline for ECG data")
    parser.add_argument("--sample_frequency", type=int,
                        help="Which sample frequency to use. Original is 1,000 Hz."
                             "Note: We can have other sample frequencies, "
                             "but then one needs to use the downsample script first",
                        default=1000)
    args = parser.parse_args()

    # load the data files
    cleaned_data_files = [f for f in os.listdir(os.path.join(CLEANED_DATA_PATH, str(args.sample_frequency))) if f.endswith('.parquet')]

    for participant_file in tqdm(cleaned_data_files):
        tqdm.write(f"Processing {participant_file.split('.')[0]}")
        participant_id = participant_file.split(".")[0]
        data_df = pd.read_parquet(os.path.join(CLEANED_DATA_PATH, str(args.sample_frequency), participant_file))

        # Load the timestamps file.
        timestamps = pd.read_csv(os.path.join(CLEANED_DATA_PATH, str(args.sample_frequency), 'Timestamps_Merged.txt'), sep="\t", decimal=".")
        timestamps['LabelStart'] = pd.to_datetime(timestamps['LabelStart'], format="%Y-%m-%d %H:%M:%S", utc=True)
        timestamps['LabelEnd'] = pd.to_datetime(timestamps['LabelEnd'], format="%Y-%m-%d %H:%M:%S", utc=True)
        timestamps["Subject_ID"] = timestamps["Subject_ID"].astype(str).str.strip()

        # Filter the timestamps for the current subject.
        timestamps_subject_id = timestamps[timestamps["Subject_ID"] == participant_id]

        # Filter data_df to include only rows within the valid experimental condition intervals.
        filtered_data_df = filter_data_by_timestamps(data_df, timestamps_subject_id)

        # Save the file now
        filtered_data_df.to_parquet(os.path.join(CLEANED_DATA_PATH, str(args.sample_frequency), f"{participant_id}.parquet"))
