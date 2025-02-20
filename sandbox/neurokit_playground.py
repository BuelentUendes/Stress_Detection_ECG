import neurokit2 as nk
import numpy as np
from pyedflib import highlevel
import pyedflib as plib

signals, signal_headers, header = highlevel.read_edf("./data/raw/Raw ECG project/30100_LAB_Conditions_ECG.edf")

ecg_signal = nk.signal_sanitize(signals[0])
ecg_signal = np.nan_to_num(ecg_signal)
cleaned_signal = nk.ecg_clean(ecg_signal, sampling_rate=1000)
cleaned_signal = np.nan_to_num(cleaned_signal)
downsampled_ecg = nk.signal_resample(cleaned_signal, sampling_rate=1000, desired_sampling_rate=100)
cleaned_signal = np.nan_to_num(downsampled_ecg)

# Add these checks after loading the signal
print(f"Signal shape: {cleaned_signal.shape}")
print(f"Signal range: {np.min(cleaned_signal)} to {np.max(cleaned_signal)}")
print(f"Number of NaN values: {np.isnan(cleaned_signal).sum()}")
print(f"Number of infinite values: {np.isinf(cleaned_signal).sum()}")

# save this downsampled_ecg:
output_path = "./data/raw/Raw ECG project/30100_LAB_Conditions_ECG_100.edf"

# Create signal header for the downsampled data
new_header = signal_headers[0].copy()
new_header['sample_rate'] = 100  # Update sample rate in header
new_header["sample_frequency"] = 100

# Write the EDF file
# signals_to_save = np.array([cleaned_signal])
# signal_headers_to_save = [new_header]
# highlevel.write_edf(output_path, signals_to_save, signal_headers_to_save, header)
#
highlevel.write_edf(
    "./data/raw/Raw ECG project/30100_LAB_Conditions_ECG_100.edf",
    np.array([downsampled_ecg]),
    [new_header],
    header
)
#
# # Step 1: Simulate an ECG signal (you can replace this with your own data)
# ecg_signal = nk.ecg_simulate(duration=10, sampling_rate=1000)  # 10 seconds, 1000 Hz
# downsampled_ecg = nk.signal_resample(ecg_signal, sampling_rate=1000, desired_sampling_rate=250)
# df, _ = nk.ecg_process(downsampled_ecg, sampling_rate=250, method='neurokit')
# # Sanitize and clean the data first before down sampling:
#
#
# original_sampling_rate = 1000  # Original sampling rate in Hz
#
# # Step 2: Define the new sampling rate
# new_sampling_rate = 250  # Desired downsampled rate in Hz
#
# # Step 3: Downsample the ECG signal
# downsampled_ecg = nk.signal_resample(ecg_signal, sampling_rate=original_sampling_rate, desired_sampling_rate=new_sampling_rate)
#
# # Step 4: Print or visualize the result
# print(f"Original Signal Length: {len(ecg_signal)}")
# print(f"Downsampled Signal Length: {len(downsampled_ecg)}")
#
# # Optional: Plot to compare (requires matplotlib)
# import matplotlib.pyplot as plt
# time_original = np.linspace(0, 100, len(ecg_signal))  # Time axis for the original signal
# time_downsampled = np.linspace(0, 100, len(downsampled_ecg))  # Time axis for the downsampled signal
#
# plt.figure(figsize=(10, 4))
# plt.plot(time_original, ecg_signal, label="Original Signal (1000 Hz)", alpha=0.8, color='blue')
# plt.plot(time_downsampled, downsampled_ecg, label="Downsampled Signal (250 Hz)", alpha=0.2, marker="x", color="orange")
# plt.legend()
# plt.title("ECG Signal Downsampling")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.show()
