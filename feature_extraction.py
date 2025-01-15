# Simple script to extract features from the cleaned data

import os
import argparse

from collections import Counter

from sia import Segmenter
from sia.io import read_csv, write_csv
from sia.segmenters import SlidingWindow

from sia.features import extract_peaks, delineate, Waves
from sia.features.time_domain import hr, hrv, time_domain, Statistic, Feature as TimeFeature
from sia.features.frequency_domain import frequency_domain, Feature as FrequencyFeature
from sia.features.nonlinear_domain import nonlinear_domain, Feature as NonlinearFeature
from sia.features.morphology_domain import morphology_domain, Feature as MorphologyFeature

from utils.helper_path import CLEANED_DATA_PATH, FEATURE_DATA_PATH
from utils.helper_functions import create_directory

#Feature Extraction Pipeline
#The pipeline defined below segments using a Sliding Window technique and calculates the features of the data.

def main(args):

    WINDOW_SIZE = args.window_size * args.sample_frequency  # how many time points we have effectively
    STEP_SIZE = int(args.window_overlap * args.sample_frequency)  # time points (units) which we shift

    input_path = os.path.join(CLEANED_DATA_PATH, str(args.sample_frequency))
    output_path = os.path.join(FEATURE_DATA_PATH, str(args.sample_frequency))
    create_directory(output_path)

    Segmenter() \
        .data(read_csv(os.path.join(input_path, '30100.csv'), columns=['ECG_Clean', 'ECG_R_Peaks', 'category'])) \
        .segment(SlidingWindow(WINDOW_SIZE, STEP_SIZE)) \
            .skip(lambda category: len(Counter(category)) > 1) \
            .extract('category', lambda category: Counter(category).most_common(1)[0][0]) \
            .use('rpeaks', lambda ECG_R_Peaks: extract_peaks(ECG_R_Peaks)) \
            .extract(hr([Statistic.MIN, Statistic.MAX, Statistic.MEAN, Statistic.STD])) \
            .extract(hrv([Statistic.MEAN, Statistic.STD, Statistic.RMS])) \
            .extract(time_domain([TimeFeature.CVNN,
                                  TimeFeature.CVSD,
                                  TimeFeature.NN20,
                                  TimeFeature.PNN20,
                                  TimeFeature.NN50,
                                  TimeFeature.PNN50])) \
            .extract(frequency_domain([FrequencyFeature.MIN, FrequencyFeature.MAX,
                                       FrequencyFeature.MEAN, FrequencyFeature.STD,
                                       FrequencyFeature.POWER, FrequencyFeature.COVARIANCE,
                                       FrequencyFeature.ENERGY, FrequencyFeature.ENTROPY])) \
            .extract(nonlinear_domain([NonlinearFeature.ENTROPY, NonlinearFeature.POINCARE,
                                       NonlinearFeature.RQA, NonlinearFeature.FRAGMENTATION])) \
            .use('tpeaks',
                 lambda ECG_Clean:
                 extract_peaks(delineate(Waves.T_Peak)(ECG_Clean))) \
            .extract(morphology_domain([MorphologyFeature.TWA])) \
        .to(write_csv(os.path.join(output_path, '[0-9]{5}.csv')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline for extracting features of the cleaned ECG data")
    parser.add_argument("--sample_frequency", type=int, default=1000, help="Sampling rate used for the dataset")
    parser.add_argument("--window_size", type=int, default=60, help="How many seconds we consider")
    parser.add_argument("--window_overlap", type=float, default=0.25,
                        help="How much shift in seconds between consecutive windows.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)


