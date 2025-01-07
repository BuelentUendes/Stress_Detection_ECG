# Simple pipeline to preprocess the data

import os
from sia import Preprocessing
from sia.io import Metadata, read_edf, write_csv
from sia.preprocessors import neurokit
import argparse

from sia.encoders import GroupEncoder
from sklearn.preprocessing import LabelEncoder

from utils.helper_path import RAW_DATA_PATH, CLEANED_DATA_PATH
from utils.helper_functions import create_directory


def main(args):

    if args.data_chunk == -1:
        input_path = os.path.join(RAW_DATA_PATH, str(args.sample_frequency))
    elif args.data_chunk == 1:
        input_path = os.path.join(RAW_DATA_PATH, str(args.sample_frequency), "part_1")
    elif args.data_chunk == 2:
        input_path = os.path.join(RAW_DATA_PATH, str(args.sample_frequency), "part_2")
    elif args.data_chunk == 3:
        input_path = os.path.join(RAW_DATA_PATH, str(args.sample_frequency), "part_3")
    elif args.data_chunk == 4:
        input_path = os.path.join(RAW_DATA_PATH, str(args.sample_frequency), "part_4")
    elif args.data_chunk == 5:
        input_path = os.path.join(RAW_DATA_PATH, str(args.sample_frequency), "part_5")

    output_path = os.path.join(CLEANED_DATA_PATH, str(args.sample_frequency))
    meta_path = os.path.join(RAW_DATA_PATH, "TimeStamps_Merged.txt")
    create_directory(output_path)

    Preprocessing(num_proc=args.number_processors) \
        .data(
            read_edf(
                os.path.join(input_path, '*.edf'),
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
        .process(neurokit(sampling_rate=args.sample_frequency, method=args.method)) \
        .filter(lambda label: [_label != "Lying_supine" for _label in label]) \
        .filter(lambda ECG_Quality: [quality > .25 for quality in ECG_Quality]) \
        .to(write_csv(os.path.join(output_path, '[0-9]{5}.csv')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing pipeline for ECG data")
    parser.add_argument("--sample_frequency", type=int,
                        help="Which sample frequency to use. Original is 1,000 Hz."
                             "Note: We can have other sample frequencies, "
                             "but then one needs to use the downsample script first",
                        default=128)
    parser.add_argument("--method", type=str, help="which method to choose for preprocessing"
                                                   "Choices: 'neurokit', 'engzeemod2012', 'elgendi2010', "
                                                   "'hamilton2002', 'pantompkins1985'",
                        default="neurokit")
    parser.add_argument("--number_processors", type=int, default=1, help="If set to -1, it uses all available")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_chunk", type=int, default=1,
                        help="Which data chunk to process. 1 for part 1, 2 for part 2, and -1 for all."
                             "Important: -1 will most likely lead to memory issues.")
    args = parser.parse_args()

    if args.number_processors == -1:
        # Get all available processors
        args.number_processors = os.cpu_count()

    main(args)

