import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from datetime import datetime

def parse_custom_timestamp(ts_str):
    main_ts, microseconds = ts_str.rsplit("-", 1)
    microseconds = microseconds[:6]
    trimmed_ts_str = f"{main_ts}-{microseconds}"
    dt_obj = datetime.strptime(trimmed_ts_str, "%Y-%m-%d_%H-%M-%S-%f")
    return dt_obj.timestamp()

def calculate_magnitude(vel):
    x = np.array([v['LinVelX'] if 'LinVelX' in v else v['AngVelX'] for v in vel])
    y = np.array([v['LinVelY'] if 'LinVelY' in v else v['AngVelY'] for v in vel])
    z = np.array([v['LinVelZ'] if 'LinVelZ' in v else v['AngVelZ'] for v in vel])
    return np.sqrt(x**2 + y**2 + z**2)

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def calculate_heart_rate(timestamps, lin_vel, ang_vel, sampling_rate, lowcut=0.75, highcut=1.6):
    heart_rates = []
    start_time = timestamps[0]
    end_time = start_time + 1  # 1-second interval

    while end_time <= timestamps[-1]:
        mask = (timestamps >= start_time) & (timestamps < end_time)
        interval_lin_vel = np.array([lin_vel[i] for i in range(len(timestamps)) if mask[i]])
        interval_ang_vel = np.array([ang_vel[i] for i in range(len(timestamps)) if mask[i]])

        if len(interval_lin_vel) > 0 and len(interval_ang_vel) > 0:
            magnitude_lin_vel = calculate_magnitude(interval_lin_vel)
            magnitude_ang_vel = calculate_magnitude(interval_ang_vel)
            combined_magnitude = magnitude_lin_vel + magnitude_ang_vel
            combined_magnitude = (combined_magnitude - np.mean(combined_magnitude)) / np.std(combined_magnitude)
            filtered_vel = apply_bandpass_filter(combined_magnitude, lowcut, highcut, sampling_rate)
            yf = fft(filtered_vel)
            xf = fftfreq(len(filtered_vel), 1 / sampling_rate)
            valid_range = (xf >= lowcut) & (xf <= highcut)
            yf_valid = yf[valid_range]
            xf_valid = xf[valid_range]

            if len(yf_valid) > 0:
                idx = np.argmax(np.abs(yf_valid))
                dominant_freq = np.abs(xf_valid[idx])
                heart_rate = dominant_freq * 60
                heart_rates.append(heart_rate)
            else:
                heart_rates.append(None)
        else:
            heart_rates.append(None)

        start_time = end_time
        end_time += 1

    return heart_rates

def calculate_and_return_heart_rates(csv_file):
    data = pd.read_csv(csv_file)
    timestamps = np.array([parse_custom_timestamp(ts) for ts in data['Timestamp']])
    lin_vel = data[['LinVelX', 'LinVelY', 'LinVelZ']].to_dict('records')
    ang_vel = data[['AngVelX', 'AngVelY', 'AngVelZ']].to_dict('records')
    sampling_rate = 37.7
    heart_rates = calculate_heart_rate(timestamps, lin_vel, ang_vel, sampling_rate)
    return heart_rates
