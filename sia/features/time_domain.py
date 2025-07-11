from typing import Union, Callable

import neurokit2 as nk

try:
    import cupy as cp
    np = cp
except ImportError:
    import numpy as np
    
from enum import Enum

def _calculate_rr_interval(rpeaks: list[int], sampling_rate: int = 1000):
    """Compute R-R intervals (also referred to as NN) in seconds"""
    rri = np.diff(rpeaks) * (1 / sampling_rate)
    return rri

class Statistic(Enum):
    MIN = lambda x: np.min(x),
    """Minimum value."""
    MAX = lambda x: np.max(x),
    """Maximum value."""
    STD = lambda x: np.std(x),
    """Standard deviation."""
    MEAN = lambda x: np.mean(x),
    """Mean value."""
    MEDIAN = lambda x: np.median(x),
    """Median value."""
    RMS = lambda x: np.sqrt(np.mean(np.diff(x) ** 2)),
    """Root mean square."""

def hr(statistics: Union[dict, list[Statistic]], sampling_rate: int = 1000):
    """Compute heart rate (HR) features.

    Parameters
    ----------
    statistics : Union[dict, list[Statistic]]
        A dictionary or list with the statistics to be computed.
    sampling_rate : int
        The sampling rate of the ECG signal.
    
    Returns
    -------
    function
        A function that computes the HR features.
    """
    def inner(rpeaks: list[int]):
        rri = _calculate_rr_interval(rpeaks, sampling_rate)
        hr = 60 / rri # HR = 60/RR interval in beats per minute
        
        result = {}
        if isinstance(statistics, dict):
            for key, value in statistics.items():
                result[f'hr_{key}'] = value(hr).item()
        elif isinstance(statistics, list):
            for statistic in statistics:
                result[f'hr_{statistic.name.lower()}'] = statistic.value[0](hr).item()
        return result
    return inner

def hrv(statistics: Union[dict, list[Statistic]], sampling_rate: int = 1000):
    """Compute heart rate variability (HRV) features.
    
    Parameters
    ----------
    statistics : Union[dict, list[Statistic]]
        A dictionary or list with the statistics to be computed.
    sampling_rate : int
        The sampling rate of the ECG signal.
        
    Returns
    -------
    function
        A function that computes the HRV features.
    """
    def inner(rpeaks: list[int]):
        hrv = np.array([(rpeaks[i]-rpeaks[i-1])/sampling_rate for i in range(1,len(rpeaks))])
        hrv *= 1000 # Convert to miliseconds

        result = {}
        if isinstance(statistics, dict):
            for key, value in statistics.items():
                result[f'hrv_{key}'] = value(hrv).item()
        elif isinstance(statistics, list):
            for statistic in statistics:
                result[f'hrv_{statistic.name.lower()}'] = statistic.value[0](hrv).item()
        return result
    return inner

class Feature(str, Enum):
    NN20 = "nn20",
    """Number of interval differences of successive RR intervals greater than 20 ms."""
    PNN20 = "pnn20",
    """Percentage of interval differences of successive RR intervals greater than 20 ms."""
    NN50 = "nn50",
    """Number of interval differences of successive RR intervals greater than 50 ms."""
    PNN50 = "pnn50",
    """Percentage of interval differences of successive RR intervals greater than 50 ms."""
    SDNN = "sdnn",
    """Standard deviation of RR intervals."""
    AVNN = "avnn",
    """Average of RR intervals."""
    CVNN = "cvnn",
    """Coefficient of variation of RR intervals"""
    CVSD = "cvsd"
    """Coefficient of variation of successive differences."""
    NK_PNN20 = "nk_pnn20"
    NK_PNN50 = "nk_pnn50"
    NK_RMSSD = "nk_rmssd"
    NK_MeanNN = "nk_mean_nn"
    NK_SDNN = "nk_sd_nn"
    NK_SD_RMSSD = "nk_sd_rmssd"
    NK_CVNN = "nk_cvnn"
    NK_CVSD = "nk_cvsd"
    NK_MEDIAN_NN = "nk_median_nn"
    NK_MAD_NN = "nk_mad_nn" # Median absolute deviation
    NK_TINN = "nk_tinn"
    NK_IQR_NN ="nk_iqr_nn"

def time_domain(features: list[Feature], sampling_rate: int = 1000):
    """Compute time domain features.

    Parameters
    ----------
    features : list[Feature]
        A list with the features to be computed.
    sampling_rate : int
        The sampling rate of the ECG signal.

    Returns
    -------
    function
        A function that computes the features in the time domain.
    """
    def inner(rpeaks: list[int]):
        rri = _calculate_rr_interval(rpeaks, sampling_rate)
        hrv_time = nk.hrv_time(rpeaks, sampling_rate=sampling_rate)

        result = {}
        for key in features:
            value = None
            if key == Feature.NN20:
                value = np.sum(np.abs(np.diff(rri)) > 0.02)
            elif key == Feature.PNN20:
                nn20 = np.sum(np.abs(np.diff(rri)) > 0.02)
                value = nn20 / len(rri)
            elif key == Feature.NN50:
                value = np.sum(np.abs(np.diff(rri)) > 0.05)
            elif key == Feature.PNN50:
                nn50 = np.sum(np.abs(np.diff(rri)) > 0.05)
                value = nn50 / len(rri)
            elif key == Feature.SDNN:
                value = np.std(rri)
            elif key == Feature.AVNN: 
                value = np.mean(rri)
            elif key == Feature.CVNN:
                value = np.std(rri) / np.mean(rri)
            elif key == Feature.CVSD:
                value = np.sqrt(np.mean(np.diff(rri) ** 2)) / np.mean(rri)
            # IMPORTANT: CHECK THIS!
            elif key == Feature.NK_PNN20:
                value = hrv_time["HRV_pNN20"].item()
            elif key == Feature.NK_PNN50:
                value = hrv_time["HRV_pNN50"].item()
            elif key == Feature.NK_RMSSD:
                value = hrv_time["HRV_RMSSD"].item()
            elif key == Feature.NK_MeanNN:
                value = hrv_time["HRV_MeanNN"].item()
            elif key == Feature.NK_SDNN:
                value = hrv_time["HRV_SDNN"].item()
            elif key == Feature.NK_SD_RMSSD:
                value = hrv_time["HRV_SDRMSSD"].item()
            elif key == Feature.NK_MEDIAN_NN:
                value = hrv_time["HRV_MedianNN"].item()
            elif key == Feature.NK_MAD_NN:
                value = hrv_time["HRV_MadNN"].item()
            elif key == Feature.NK_TINN:
                value = hrv_time["HRV_TINN"].item()
            elif key == Feature.NK_CVNN:
                value = hrv_time["HRV_CVNN"].item()
            elif key == Feature.NK_CVSD:
                value = hrv_time["HRV_CVSD"].item()
            elif key == Feature.NK_IQR_NN:
                value = hrv_time["HRV_IQRNN"].item()
            if value != None:
                try:
                    result[key] = value.item()
                except AttributeError:
                    result[key] = value
        return result
    return inner