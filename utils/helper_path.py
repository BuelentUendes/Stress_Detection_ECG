# Helper file for used paths
import os
from os.path import abspath

# Define the common paths
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = abspath(os.path.join(FILE_PATH, './../'))
DATA_PATH = abspath(os.path.join(BASE_PATH, './', "data"))
CLEANED_DATA_PATH = abspath(os.path.join(DATA_PATH, "cleaned"))
FEATURE_DATA_PATH = abspath(os.path.join(DATA_PATH, "features"))
RAW_DATA_PATH = abspath(os.path.join(DATA_PATH, "raw", "Raw ECG project"))
MODELS_PATH = abspath(os.path.join(BASE_PATH, './', "models"))
CONFIG_PATH = abspath(os.path.join(BASE_PATH, './', "configs"))
RESULTS_PATH = abspath(os.path.join(BASE_PATH, './', "results"))
