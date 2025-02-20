# Load NeuroKit and other useful packages
import neurokit2 as nk
import numpy as np
import pandas as pd

# Retrieve ECG data from data folder
ecg_signal = nk.data(dataset="ecg_1000hz")
# Extract R-peaks locations
_, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=1000)

# Delineate the ECG signal
_, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=1000, method="peak")
_, waves_peak_2 = nk.ecg_delineate(ecg_signal, sampling_rate=1000, method="peak")
_, waves_peak_3 = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=1000, method="dwt")


print(f"What the fuck is the difference?")