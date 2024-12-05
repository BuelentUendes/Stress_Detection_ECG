# Simple pipeline to preprocess the data

import os
from sia import Preprocessing
from sia.io import Metadata, read_edf, read_csv, write_csv
from sia.preprocessors import neurokit
import argparse

from sia.encoders import GroupEncoder
from sklearn.preprocessing import LabelEncoder

from utils.helper_path import RAW_DATA_PATH, CLEANED_DATA_PATH


def create_directory(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        path: Path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def main(args):

    input_path = os.path.join(RAW_DATA_PATH, str(args.sample_frequency))
    output_path = os.path.join(CLEANED_DATA_PATH, str(args.sample_frequency))
    meta_path = os.path.join(RAW_DATA_PATH, "TimeStamps_Merged.txt")
    create_directory(output_path)

    # Get some safeguard
    if args.sample_frequency == 1000 and not args.clean_before_processing:
        print("WARNING: The original dataset of 1000HZ is not cleaned! So it is advisable to set the flag!")

    Preprocessing() \
        .data(
            read_edf(
                os.path.join(input_path, '30100_LAB_Conditions_ECG.edf'),
                Metadata(meta_path).on_regex(r'[0-9]{5}'),
                sampling_rate=args.sample_frequency,
            )
        ) \
        .rename({'category': 'label'}) \
        .encode({'label': 'category'}, GroupEncoder({
            'baseline': ['Sitting', 'Recov1', 'Recov2', 'Recov3', 'Recov4', 'Recov5', 'Recov6'],
            'mental_stress': ['TA', 'SSST_Sing_countdown', 'Pasat', 'Raven', 'TA_repeat', 'Pasat_repeat'],
            'high_physical_activity': ['Treadmill1', 'Treadmill2', 'Treadmill3', 'Treadmill4', 'Walking_fast_pace',
                                       'Cycling', 'stairs_up_and_down'],
            'moderate_physical_activity': ['Walking_own_pace', 'Dishes', 'Vacuum'],
            'low_physical_activity': ['Standing', 'Recov_standing', 'Lying_supine']     # Updated low physical activity
        })) \
        .filter(lambda category: [_category != None for _category in category]) \
        .process(neurokit(sampling_rate=args.sample_frequency, clean_before_processing=True)) \
        .filter(lambda label: [_label != "Lying_supine" for _label in label]) \
        .filter(lambda ECG_Quality: [quality > .25 for quality in ECG_Quality]) \
        .to(write_csv(os.path.join(output_path, '[0-9]{5}.csv')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing pipeline for ECG data")
    parser.add_argument("--sample_frequency", type=int,
                        help="Which sample frequency to use. Original is 1,000 Hz."
                             "Note: We can have other sample frequencies, "
                             "but then one needs to use the downsample script first",
                        default=1000)
    parser.add_argument("--clean_before_processing", action="store_true",
                        help="If we should clean the signal before processing. NOTE: 1000HZ needs to be cleaned!")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)


