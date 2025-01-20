import warnings
from warnings import warn

try:
    import cupy as cp
    np = cp
except ImportError:
    import numpy as np
    
from enum import Enum

class Feature(str, Enum):
    TWA = "twa"
    """T-wave alternans (TWA) feature."""

def morphology_domain(features: tuple[Feature]):
    """Compute morphology domain features.

    Parameters
    ----------
    features : tuple[Feature]
        A tuple with the features to be computed.

    Returns
    -------
    function
        A function that computes the features in the morphology domain.    
    """
    def inner(ECG_Clean: list[float], tpeaks: list[int]):
        result = {}
        warnings.filterwarnings("ignore")
        for feature in features:
            if feature == Feature.TWA:
                twa = calculate_twa(ECG_Clean, tpeaks)
                result.update({ "twa": twa.item() })
            else:
                raise ValueError(f"Feature {feature} is not valid.")
        warnings.filterwarnings("default")
        return result
    return inner

def calculate_twa(signal: list[float], tpeaks: list[int]):
    """Compute the T-wave alternans (TWA) feature.

    Parameters
    ----------
    signal : list[float]
        The ECG signal.
    tpeaks : list[int]
        The T-wave peaks.
        
    Returns
    -------
    dict
        A dictionary containing the TWA feature.
    """
    # Check if signal is empty
    if signal is None or len(signal) == 0:
        return np.nan

    # Convert to numpy array and check if conversion was successful
    try:
        signal = np.array(signal)
        if signal.size == 0:
            warn("Empty signal array", RuntimeWarning)
            return np.nan

    except Exception as e:
        warn(f"Failed to convert signal to numpy array: {str(e)}", RuntimeWarning)
        return np.nan

    if len(tpeaks) < 2:
        warn("Insufficient T-peaks to calculate TWA (minimum 2 required)", RuntimeWarning)
        return np.nan

    # Ensure T-peaks are within signal bounds
    valid_tpeaks = [t for t in tpeaks if 0 <= t < len(signal)]
    if len(valid_tpeaks) < 2:
        warn("No valid T-peaks within signal bounds", RuntimeWarning)
        return np.nan

    # Divide the T-peaks into two buckets, even and odd.
    even_bucket = tpeaks[1::2]
    odd_bucket = tpeaks[::2]

    # Calculate the average of the even and odd buckets.
    average_t_even = np.mean(np.take(signal, even_bucket))
    average_t_odd = np.mean(np.take(signal, odd_bucket))

    if average_t_even is None or average_t_odd is None:
        return np.nan
    else:
        # Calculate the difference in amplitude between the even and odd buckets.
        twa = abs(average_t_even - average_t_odd)
        return twa