import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


import mne
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

# IIR滤波器
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# 应用滤波器
def preprocess_raw(raw, lowcut=0.5, highcut=45):
    raw_filtered = raw.copy().filter(l_freq=lowcut, h_freq=highcut, fir_design='firwin')
    return raw_filtered

# 将 RawArray 对象转换为数据矩阵
def raw_to_matrix(raw):
    data, times = raw.get_data(return_times=True)
    return data.T

# 数据增强
def augment_data(data, labels, augmentation_factor=5):
    augmented_data = np.tile(data, (augmentation_factor, 1, 1))
    augmented_labels = np.tile(labels, augmentation_factor)
    return augmented_data, augmented_labels

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

    # Correct the attribute from 'SSvalues' to 'values'
    data1 = data.iloc[:, 1:].transpose().values

    ch_types = ['eeg'] * len(ch_names)
    sfreq = 100  # Hz

    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(data1, info)

    # Create a custom montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)

    return raw


# 读取和预处理数据
def preprocess_data(files):
    raws = [preprocess_raw(create_raw_from_csv(file)) for file in files]
    data = np.array([raw_to_matrix(raw) for raw in raws])
    return data

# 数据生成器
def data_generator(files, labels, batch_size):
    while True:
        for start in range(0, len(files), batch_size):
            end = min(start + batch_size, len(files))
            batch_files = files[start:end]
            batch_labels = labels[start:end]
            batch_data = preprocess_data(batch_files)
            yield batch_data, to_categorical(batch_labels, 2)

# 读取文件路径
train_awake_files = [
    'E:/臻泰实习/深度学习论文/Dataset/Dataset/Train/Awake/P2_afternoon.csv'

]

train_drowsy_files = [
    'E:/臻泰实习/深度学习论文/Dataset/Dataset/Train/Drowsy/P2_evening.csv'

]

test_awake_files = [
    'E:/臻泰实习/深度学习论文/Dataset/Dataset/Test/Awake/P8_evening.csv'
]

test_drowsy_files = [
    'E:/臻泰实习/深度学习论文/Dataset/Dataset/Test/Drowsy/P3_afternoon.csv'
]

# 创建标签
train_awake_labels = np.zeros(len(train_awake_files))
train_drowsy_labels = np.ones(len(train_drowsy_files))
test_awake_labels = np.zeros(len(test_awake_files))
test_drowsy_labels = np.ones(len(test_drowsy_files))

# 合并训练和测试数据
train_files = train_awake_files + train_drowsy_files
train_labels = np.concatenate((train_awake_labels, train_drowsy_labels))
test_files = test_awake_files + test_drowsy_files
test_labels = np.concatenate((test_awake_labels, test_drowsy_labels))

# 生成器
batch_size = 1
train_generator = data_generator(train_files, train_labels, batch_size)

# 预处理测试数据
test_data = preprocess_data(test_files)
test_labels = to_categorical(test_labels, 2)

# 构建CNN模型
def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, 16, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv1D(64, 16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(2, activation='softmax'))

    return model

input_shape = (115200, 2)
model = build_cnn_model(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
steps_per_epoch = len(train_files) // batch_size
history = model.fit(train_generator, epochs=15, steps_per_epoch=steps_per_epoch, validation_data=(test_data, test_labels))

# 评估模型
loss, accuracy = model.evaluate(test_data, test_labels)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

import matplotlib.pyplot as plt

# 绘制训练和验证的准确率和损失
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_history(history)
