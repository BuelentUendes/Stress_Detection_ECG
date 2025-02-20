import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
from neurokit2.hrv.hrv_utils import _hrv_format_input
from neurokit2.complexity import fractal_dfa
from neurokit2.hrv.hrv_nonlinear import _hrv_dfa

data = nk.data("bio_resting_5min_100hz")
data.head()


def get_signal_windows(signal, window_seconds, sampling_rate, overlap_percent=0):
    """
    Split a signal into windows of specified duration.

    Args:
        signal: Input signal array
        window_seconds: Window size in seconds
        sampling_rate: Sampling rate in Hz
        overlap_percent: Overlap between windows (0-100), default 0

    Returns:
        numpy array of windowed segments
    """
    # Calculate samples per window
    window_samples = int(window_seconds * sampling_rate)

    # Calculate step size based on overlap
    step_size = int(window_samples * (1 - overlap_percent / 100))

    # Calculate number of windows
    n_windows = (len(signal) - window_samples) // step_size + 1

    # Create windowed array
    windows = np.zeros((n_windows, window_samples))

    # Fill windows
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_samples
        windows[i] = signal[start_idx:end_idx]

    return windows

ecg_cleaned = nk.ecg_clean(data["ECG"], sampling_rate=100)
peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=100)
heart_rate = nk.signal_rate(info, sampling_rate=100, desired_length=len(ecg_cleaned))

# Replace the manual windowing with the function
ecg_windowed = get_signal_windows(ecg_cleaned, window_seconds=30, sampling_rate=100, overlap_percent=0.5)

for window in ecg_windowed:
    # Compute HRC indices
    peaks, info = nk.ecg_peaks(window, sampling_rate=100)
    # Calculate the rri intervals
    rri, rri_time, rri_missing = _hrv_format_input(peaks, sampling_rate=100)
    # Empty result dictionary
    out = {}
    out = _hrv_dfa(rri, out)


# Get for each of the window segments now the quality of the ECG signal
ecg_windowed_quality = [nk.ecg_quality(ecg_window, sampling_rate=100) for ecg_window in ecg_windowed]
ecg_windowed_quality_zhao = [nk.ecg_quality(ecg_window, sampling_rate=100, method="zhao2018") for ecg_window in ecg_windowed]




# Calculate now the heart rate features per window
def get_heart_rate_features(signal, sampling_rate):

    for window in signal:
        peaks, info = nk.hrv_time(window, sampling_rate=sampling_rate)




# print(ecg_windowed.shape[1])
# # Plot the one with very low score
# x_line = np.arange(0, ecg_windowed.shape[1])
# plt.plot(x_line, ecg_windowed[2], color="red", label="low quality")
# plt.plot(x_line, ecg_windowed[7], color="blue", label="good quality")
# plt.legend()
# plt.show()

