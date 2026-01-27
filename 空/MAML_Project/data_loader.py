"""
Few-Shot 数据加载器
支持 N-way K-shot 任务采样
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
import random


class FewShotTaskSampler:
    """
    Few-Shot 任务采样器

    参数:
        data: 数据 [num_samples, ...]
        labels: 标签 [num_samples]
        n_way: 每个任务的类别数
        k_shot: 每个类别的支持集样本数
        q_query: 每个类别的查询集样本数
    """

    def __init__(
            self,
            data: np.ndarray,
            labels: np.ndarray,
            n_way: int = 5,
            k_shot: int = 5,
            q_query: int = 15
    ):
        self.data = data
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        # 按类别组织数据
        self.class_indices = {}
        unique_labels = np.unique(labels)

        for label in unique_labels:
            self.class_indices[label] = np.where(labels == label)[0]

        self.classes = list(self.class_indices.keys())

        print(f"📊 数据集统计:")
        print(f"   总样本数: {len(data)}")
        print(f"   类别数: {len(self.classes)}")
        print(f"   各类别样本数: {[len(self.class_indices[c]) for c in self.classes]}")

    def sample_task(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样一个 N-way K-shot 任务

        返回:
            support_x: [N_way * K_shot, ...]
            support_y: [N_way * K_shot]
            query_x: [N_way * Q_query, ...]
            query_y: [N_way * Q_query]
        """
        # 随机选择 N 个类别
        selected_classes = random.sample(self.classes, self.n_way)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for new_label, original_class in enumerate(selected_classes):
            # 获取该类别的所有索引
            class_indices = self.class_indices[original_class]

            # 随机选择 K + Q 个样本
            selected_indices = random.sample(
                list(class_indices),
                self.k_shot + self.q_query
            )

            # 分割为支持集和查询集
            support_indices = selected_indices[:self.k_shot]
            query_indices = selected_indices[self.k_shot:]

            # 添加到列表
            for idx in support_indices:
                support_x.append(self.data[idx])
                support_y.append(new_label)  # 重新映射标签

            for idx in query_indices:
                query_x.append(self.data[idx])
                query_y.append(new_label)

        # 转换为 Tensor
        support_x = torch.FloatTensor(np.array(support_x))
        support_y = torch.LongTensor(support_y)
        query_x = torch.FloatTensor(np.array(query_x))
        query_y = torch.LongTensor(query_y)

        # 打乱顺序
        support_perm = torch.randperm(len(support_y))
        query_perm = torch.randperm(len(query_y))

        return (
            support_x[support_perm],
            support_y[support_perm],
            query_x[query_perm],
            query_y[query_perm]
        )

    def sample_batch(self, batch_size: int) -> List[Tuple]:
        """采样一批任务"""
        return [self.sample_task() for _ in range(batch_size)]


class SyntheticTimeSeriesDataset:
    """
    合成时间序列数据集（用于测试和演示）

    生成不同模式的时间序列用于分类
    """

    def __init__(
            self,
            num_classes: int = 10,
            samples_per_class: int = 100,
            seq_length: int = 128,
            noise_level: float = 0.1
    ):
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.seq_length = seq_length

        print(f"\n🔧 生成合成时间序列数据...")

        data = []
        labels = []

        for class_idx in range(num_classes):
            class_data = self._generate_class_data(class_idx, samples_per_class, noise_level)
            data.extend(class_data)
            labels.extend([class_idx] * samples_per_class)

        self.data = np.array(data)
        self.labels = np.array(labels)

        print(f"   ✅ 生成完成: {self.data.shape}")

    def _generate_class_data(self, class_idx: int, n_samples: int, noise: float) -> List[np.ndarray]:
        """生成特定类别的数据"""
        t = np.linspace(0, 4 * np.pi, self.seq_length)
        samples = []

        for _ in range(n_samples):
            # 不同类别使用不同的波形模式
            if class_idx % 5 == 0:
                # 正弦波 + 不同频率
                freq = 1 + class_idx * 0.3
                signal = np.sin(freq * t)
            elif class_idx % 5 == 1:
                # 锯齿波
                signal = 2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi) + 0.5))
            elif class_idx % 5 == 2:
                # 方波
                signal = np.sign(np.sin(t * (1 + class_idx * 0.2)))
            elif class_idx % 5 == 3:
                # 复合波
                signal = np.sin(t) + 0.5 * np.sin(2 * t + class_idx)
            else:
                # 阻尼振荡
                signal = np.exp(-0.1 * t) * np.sin(t * (1 + class_idx * 0.1))

            # 添加类别特定的偏移和缩放
            signal = signal * (1 + class_idx * 0.1) + class_idx * 0.2

            # 添加噪声
            signal += np.random.randn(self.seq_length) * noise

            samples.append(signal.astype(np.float32))

        return samples

    def get_task_sampler(self, n_way: int, k_shot: int, q_query: int) -> FewShotTaskSampler:
        """获取任务采样器"""
        return FewShotTaskSampler(
            data=self.data,
            labels=self.labels,
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query
        )


class SyntheticEEGDataset:
    """
    合成 EEG 数据集（用于测试）
    模拟多通道脑电信号
    """

    def __init__(
            self,
            num_classes: int = 4,
            samples_per_class: int = 200,
            num_channels: int = 32,
            seq_length: int = 256,
            fs: float = 256.0  # 采样率
    ):
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.seq_length = seq_length

        print(f"\n🧠 生成合成 EEG 数据...")

        data = []
        labels = []

        for class_idx in range(num_classes):
            class_data = self._generate_eeg_class(class_idx, samples_per_class, fs)
            data.extend(class_data)
            labels.extend([class_idx] * samples_per_class)

        self.data = np.array(data)
        self.labels = np.array(labels)

        print(f"   ✅ 生成完成: {self.data.shape}")

    def _generate_eeg_class(self, class_idx: int, n_samples: int, fs: float) -> List[np.ndarray]:
        """生成模拟不同脑电状态的数据"""
        t = np.arange(self.seq_length) / fs
        samples = []

        # 不同类别代表不同脑电状态
        # 0: Alpha 波 (8-13 Hz) - 放松状态
        # 1: Beta 波 (13-30 Hz) - 活跃思考
        # 2: Theta 波 (4-8 Hz) - 困倦状态
        # 3: 混合波 - 正常清醒

        freq_bands = {
            0: (8, 13),  # Alpha
            1: (13, 30),  # Beta
            2: (4, 8),  # Theta
            3: (8, 30),  # Mixed
        }

        band = freq_bands.get(class_idx % 4, (8, 13))

        for _ in range(n_samples):
            eeg_signal = np.zeros((self.num_channels, self.seq_length), dtype=np.float32)

            for ch in range(self.num_channels):
                # 主频率成分
                main_freq = np.random.uniform(band[0], band[1])
                signal = np.sin(2 * np.pi * main_freq * t)

                # 添加谐波
                for harmonic in range(2, 4):
                    freq = main_freq * harmonic + np.random.randn() * 0.5
                    amp = 1.0 / harmonic
                    signal += amp * np.sin(2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)

                # 通道间相关性
                if ch > 0:
                    signal += 0.3 * eeg_signal[ch - 1]

                # 添加噪声
                signal += np.random.randn(self.seq_length) * 0.2

                eeg_signal[ch] = signal

            samples.append(eeg_signal)

        return samples

    def get_task_sampler(self, n_way: int, k_shot: int, q_query: int) -> FewShotTaskSampler:
        """获取任务采样器"""
        return FewShotTaskSampler(
            data=self.data,
            labels=self.labels,
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query
        )


def load_custom_dataset(
        data_path: str,
        label_path: str,
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 15
) -> FewShotTaskSampler:
    """
    加载自定义数据集

    参数:
        data_path: 数据文件路径 (.npy)
        label_path: 标签文件路径 (.npy)
    """
    print(f"\n📂 加载自定义数据集...")
    print(f"   数据路径: {data_path}")
    print(f"   标签路径: {label_path}")

    data = np.load(data_path)
    labels = np.load(label_path)

    return FewShotTaskSampler(
        data=data,
        labels=labels,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query
    )
