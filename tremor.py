import numpy as np
import scipy.signal as signal
import json

# Function to read hand motion data from a JSON file
def read_hand_data(file_path):
    with open(file_path, 'r') as file:
        data_record = json.load(file)
    return data_record

# Function to extract hand motion data
def extract_hand_data(record):
    left_wrist_positions = [rec['leftWristPosition'] for rec in record['headControllersMotionRecordList']]
    return np.array([[pos['x'], pos['y'], pos['z']] for pos in left_wrist_positions])

# Function to filter the signal
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

# Function to perform frequency analysis
def frequency_analysis(data, fs):
    n = len(data)
    freqs = np.fft.fftfreq(n, 1/fs)
    fft_result = np.fft.fft(data)
    return freqs, np.abs(fft_result)

# Main function to run the analysis
def main(file_path, fs=100.0):
    # Step 1: Read the hand motion data from the specified file
    data_record = read_hand_data(file_path)
    
    # Step 2: Extract hand motion data
    hand_data = extract_hand_data(data_record)
    
    # Step 3: Extract the x, y, and z components of wrist positions
    wrist_x = hand_data[:, 0]  # X-axis
    wrist_y = hand_data[:, 1]  # Y-axis
    wrist_z = hand_data[:, 2]  # Z-axis

    # Step 4: Calculate the mean position for baseline
    baseline_x = np.mean(wrist_x)
    baseline_y = np.mean(wrist_y)
    baseline_z = np.mean(wrist_z)

    # Step 5: Calculate deviations from the baseline
    deviations_x = wrist_x - baseline_x
    deviations_y = wrist_y - baseline_y
    deviations_z = wrist_z - baseline_z

    # Step 6: Combine deviations for analysis
    combined_deviations = deviations_x + deviations_y + deviations_z
    
    # Step 7: Apply bandpass filter to combined deviations
    filtered_deviations = bandpass_filter(combined_deviations, 4, 8, fs)  # 4-8 Hz for detection
    
    # Step 8: Frequency analysis on the filtered deviations
    freqs, fft_values = frequency_analysis(filtered_deviations, fs)
    
    # Step 9: Identify significant frequencies (greater than a threshold)
    threshold = np.max(fft_values) * 0.1  # 10% of maximum for significant peaks
    significant_frequencies = freqs[np.where(fft_values > threshold)]
    
    # Step 10: Check for Parkinson's risk based on significant frequencies
    has_parkinsons_risk = any((freq >= 4) and (freq <= 7) for freq in significant_frequencies)

    return has_parkinsons_risk

# Example file path
file_path = 'path_to_your_data_file.json'

# Run the main analysis and print the output
has_risk = main(file_path)
