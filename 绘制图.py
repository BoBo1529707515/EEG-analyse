import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import butter, lfilter
import mne


# 读取 CSV 文件并转换为 RawArray
def create_raw_from_csv(file_path):
    data = pd.read_csv(file_path)
    ch_names = data.columns[1:].tolist()  # Skip the Timestamp column

    # Rename columns to standard 10-20 names
    rename_dict = {
        'EEG.O1': 'O1',
        'EEG.O2': 'O2',
        # Add other renaming if needed
    }
    ch_names = [rename_dict.get(ch, ch) for ch in ch_names]

    data1 = data.iloc[:, 1:].transpose().values

    ch_types = ['eeg'] * len(ch_names)
    sfreq = 100  # Hz

    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(data1, info)

    # Create a custom montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)

    return raw


# 时域特征提取
def time_domain_features(data):
    features = {}
    features['mean'] = np.mean(data, axis=1)
    features['energy'] = np.sum(data ** 2, axis=1)
    features['variance'] = np.var(data, axis=1)
    features['rms'] = np.sqrt(np.mean(data ** 2, axis=1))
    return features


# 频域特征提取
def frequency_domain_features(data, sfreq):
    freqs = np.fft.fftfreq(data.shape[1], 1 / sfreq)
    fft_vals = np.abs(np.fft.fft(data, axis=1))

    features = {}
    features['delta'] = np.mean(fft_vals[:, (freqs >= 0.5) & (freqs <= 4)], axis=1)
    features['theta'] = np.mean(fft_vals[:, (freqs >= 4) & (freqs <= 8)], axis=1)
    features['alpha'] = np.mean(fft_vals[:, (freqs >= 8) & (freqs <= 13)], axis=1)
    features['beta'] = np.mean(fft_vals[:, (freqs >= 13) & (freqs <= 30)], axis=1)
    features['gamma'] = np.mean(fft_vals[:, (freqs >= 30) & (freqs <= 45)], axis=1)

    return features


# Higuchi分形维数计算
def higuchi_fd(data, kmax=10):
    L = []
    x = np.array(data)
    N = len(x)
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            Lmk = sum(abs(x[m + i * k] - x[m + (i - 1) * k]) for i in range(1, int((N - m) / k)))
            Lmk = (Lmk * (N - 1) / (int((N - m) / k) * k)) / k
            Lk.append(Lmk)
        L.append(np.mean(Lk))
    L = np.log(L)
    ln_k = np.log(range(1, kmax + 1))
    higuchi, _ = np.polyfit(ln_k, L, 1)
    return higuchi


# Hjorth参数计算
def hjorth_parameters(data):
    diff_input = np.diff(data)
    diff_diff = np.diff(diff_input)

    var_zero = np.var(data)
    var_d1 = np.var(diff_input)
    var_d2 = np.var(diff_diff)

    activity = var_zero
    mobility = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / mobility

    return activity, mobility, complexity


# 绘制时域和频域图像
def plot_eeg_features(raw, time_features, freq_features):
    data, times = raw.get_data(return_times=True)

    plt.figure(figsize=(15, 10))

    # 时域图像
    plt.subplot(2, 1, 1)
    plt.plot(times, data.T)
    plt.xlabel('Time (s)')
    plt.ylabel('EEG Signal')
    plt.title('Time Domain Signal')

    # 频域图像
    plt.subplot(2, 1, 2)
    freqs = np.fft.fftfreq(data.shape[1], 1 / raw.info['sfreq'])
    fft_vals = np.abs(np.fft.fft(data, axis=1))
    plt.plot(freqs[:len(freqs) // 2], fft_vals[:, :len(freqs) // 2].T)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Domain Signal')

    plt.tight_layout()
    plt.show()


# 绘制各个波段的频谱图
def plot_band_spectra(raw, freq_features):
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    freqs = np.fft.fftfreq(data.shape[1], 1 / sfreq)
    fft_vals = np.abs(np.fft.fft(data, axis=1))

    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }

    plt.figure(figsize=(15, 10))
    for i, (band, (low, high)) in enumerate(bands.items()):
        plt.subplot(len(bands), 1, i + 1)
        band_mask = (freqs >= low) & (freqs <= high)
        plt.plot(freqs[band_mask], fft_vals[:, band_mask].T)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title(f'{band} Band ({low}-{high} Hz)')

    plt.tight_layout()
    plt.show()


# 示例文件路径
file_path = r'E:\臻泰实习\深度学习论文\Dataset\Dataset\Train\Drowsy\P1_afternoon.csv'

# 读取和预处理数据
raw = create_raw_from_csv(file_path)
data = raw.get_data()

# 提取特征
time_features = time_domain_features(data)
freq_features = frequency_domain_features(data, raw.info['sfreq'])

# 计算Higuchi分形维数和Hjorth参数
higuchi_features = np.apply_along_axis(higuchi_fd, 1, data)
hjorth_activities, hjorth_mobilities, hjorth_complexities = np.apply_along_axis(hjorth_parameters, 1, data).T

# 输出特征值
print("Time Domain Features:", time_features)
print("Frequency Domain Features:", freq_features)
print("Higuchi Fractal Dimensions:", higuchi_features)
print("Hjorth Parameters (Activity, Mobility, Complexity):")
print("Activities:", hjorth_activities)
print("Mobilities:", hjorth_mobilities)
print("Complexities:", hjorth_complexities)

# 绘制特征图像
plot_eeg_features(raw, time_features, freq_features)
plot_band_spectra(raw, freq_features)
