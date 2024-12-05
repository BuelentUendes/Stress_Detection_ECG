import warnings
import neurokit2 as nk
from neurokit2 import ecg_peaks, signal_rate, ecg_quality, ecg_delineate, ecg_phase
import pandas as pd

from typing import Callable


def preprocessing_pipeline(ecg_cleaned, sampling_rate: int = 1000, method: str = "neurokit"):
    """
    Runs through the preprocessing pipeline for an already cleaned signal
    :param signal: Cleaned ECG signal
    :return: pd dataframe
    """

    # Detect R-peaks
    instant_peaks, info = ecg_peaks(
        ecg_cleaned=ecg_cleaned,
        sampling_rate=sampling_rate,
        method=method,
        correct_artifacts=True,
    )

    # Calculate heart rate
    rate = signal_rate(
        info, sampling_rate=sampling_rate, desired_length=len(ecg_cleaned)
    )

    # Assess signal quality
    quality = ecg_quality(
        ecg_cleaned, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate
    )

    # Merge signals in a DataFrame
    signals = pd.DataFrame(
        {
            "ECG_Clean": ecg_cleaned,
            "ECG_Rate": rate,
            "ECG_Quality": quality,
        }
    )

    # Delineate QRS complex
    delineate_signal, delineate_info = ecg_delineate(
        ecg_cleaned=ecg_cleaned, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate
    )
    info.update(delineate_info)  # Merge waves indices dict with info dict

    # Determine cardiac phases
    cardiac_phase = ecg_phase(
        ecg_cleaned=ecg_cleaned,
        rpeaks=info["ECG_R_Peaks"],
        delineate_info=delineate_info,
    )

    # Add additional information to signals DataFrame
    signals = pd.concat(
        [signals, instant_peaks, delineate_signal, cardiac_phase], axis=1
    )

    # return signals DataFrame and R-peak locations
    return signals, info


def pantompkins(sampling_rate: int = 1000) -> Callable[[list], dict]:
    """Compute ECG features using the Pan-Tompkins algorithm.

    Parameters
    ----------
    sampling_rate : int, optional
        The sampling rate of the ECG signal, by default 1000.
    
    Returns
    -------
    function
        A function that computes the ECG features using the Pan-Tompkins algorithm.
    """
    def inner(signal: list[float]): 
        warnings.filterwarnings("ignore")
        df, _ = nk.ecg_process(signal, sampling_rate=sampling_rate, method='pantompkins1985')
        warnings.filterwarnings("default")
        return df.to_dict('list')
    return inner

def hamilton(sampling_rate: int = 1000) -> Callable[[list], dict]:
    """Compute ECG features using the Hamilton algorithm.

    Parameters
    ----------
    sampling_rate : int, optional
        The sampling rate of the ECG signal, by default 1000.

    Returns
    -------
    function
        A function that computes the ECG features using the Hamilton algorithm.
    """
    def inner(signal: list[float]): 
        warnings.filterwarnings("ignore")
        df, _ = nk.ecg_process(signal, sampling_rate=sampling_rate, method='hamilton2002')
        warnings.filterwarnings("default")
        return df.to_dict('list')
    return inner

def elgendi(sampling_rate: int = 1000) -> Callable[[list], dict]:
    """Compute ECG features using the Elgendi algorithm.

    Parameters
    ----------
    sampling_rate : int, optional
        The sampling rate of the ECG signal, by default 1000.

    Returns
    -------
    function
        A function that computes the ECG features using the Elgendi algorithm.
    """
    def inner(signal: list[float]): 
        warnings.filterwarnings("ignore")
        df, _ = nk.ecg_process(signal, sampling_rate=sampling_rate, method='elgendi2010')
        warnings.filterwarnings("default")
        return df.to_dict('list')
    return inner

def engzeemod(sampling_rate: int = 1000) -> Callable[[list], dict]:
    """Compute ECG features using the Engzee Modified algorithm.

    Parameters
    ----------
    sampling_rate : int, optional
        The sampling rate of the ECG signal, by default 1000.

    Returns
    -------
    function
        A function that computes the ECG features using the Engzee Modified algorithm.
    """
    def inner(signal: list[float]): 
        warnings.filterwarnings("ignore")
        df, _ = nk.ecg_process(signal, sampling_rate=sampling_rate, method='engzeemod2012')
        warnings.filterwarnings("default")
        return df.to_dict('list')
    return inner

def neurokit(sampling_rate: int = 1000, clean_before_processing: bool = True) -> Callable[[list], dict]:
    """Compute ECG features using the NeuroKit algorithm.

    Parameters
    ----------
    sampling_rate : int, optional
        The sampling rate of the ECG signal, by default 1000.
    clean_before_processing : bool, optional
        If the signal needs to get cleaned before processing.
        In our dataset, the original sample frequency is at 1000 Hz which is not yet cleaned.

    Returns
    -------
    function
        A function that computes the ECG features using the NeuroKit algorithm.
    """
    def inner(signal: list[float]): 
        warnings.filterwarnings("ignore")
        if clean_before_processing:
            df, _ = nk.ecg_process(signal, sampling_rate=sampling_rate, method='neurokit')
        else:
            df, _ = preprocessing_pipeline(
                signal,
                sampling_rate=sampling_rate,
                method="neurokit"
            )

        warnings.filterwarnings("default")
        return df.to_dict('list')
    return inner