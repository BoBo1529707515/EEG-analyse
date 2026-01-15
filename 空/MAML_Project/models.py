"""
MAML 基础模型定义
支持时间序列和EEG信号分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """卷积块"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels, momentum=1, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class TimeSeriesMAMLNet(nn.Module):
    """
    用于时间序列分类的 MAML 网络

    参数:
        input_channels: 输入通道数 (如 EEG 的电极数)
        seq_length: 序列长度
        num_classes: 分类类别数
        hidden_dim: 隐藏层维度
    """

    def __init__(
            self,
            input_channels: int = 1,
            seq_length: int = 128,
            num_classes: int = 5,
            hidden_dim: int = 64
    ):
        super().__init__()

        self.input_channels = input_channels
        self.seq_length = seq_length

        # 1D 卷积特征提取器
        self.features = nn.Sequential(
            ConvBlock(input_channels, hidden_dim),
            nn.MaxPool1d(2),
            ConvBlock(hidden_dim, hidden_dim),
            nn.MaxPool1d(2),
            ConvBlock(hidden_dim, hidden_dim),
            nn.MaxPool1d(2),
            ConvBlock(hidden_dim, hidden_dim),
            nn.AdaptiveAvgPool1d(1)
        )

        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        前向传播
        x: [batch, channels, seq_length] 或 [batch, seq_length]
        """
        # 确保输入是3D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, seq_length]

        # 特征提取
        features = self.features(x)  # [batch, hidden_dim, 1]
        features = features.squeeze(-1)  # [batch, hidden_dim]

        # 分类
        logits = self.classifier(features)

        return logits


class EEGMAMLNet(nn.Module):
    """
    用于 EEG 信号分类的 MAML 网络

    参数:
        num_channels: EEG 通道数 (电极数)
        seq_length: 时间点数
        num_classes: 分类类别数
    """

    def __init__(
            self,
            num_channels: int = 64,
            seq_length: int = 256,
            num_classes: int = 4,
            hidden_dim: int = 128
    ):
        super().__init__()

        # 空间卷积 (跨通道)
        self.spatial_conv = nn.Conv2d(1, 32, kernel_size=(num_channels, 1))
        self.spatial_bn = nn.BatchNorm2d(32)

        # 时间卷积
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        x: [batch, channels, time] - EEG 数据
        """
        # 添加维度 [batch, 1, channels, time]
        x = x.unsqueeze(1)

        # 空间卷积
        x = self.spatial_conv(x)  # [batch, 32, 1, time]
        x = self.spatial_bn(x)
        x = F.relu(x)
        x = x.squeeze(2)  # [batch, 32, time]

        # 时间卷积
        x = self.temporal_conv(x)  # [batch, hidden_dim, 1]
        x = x.squeeze(-1)  # [batch, hidden_dim]

        # 分类
        logits = self.classifier(x)

        return logits


class LSTMMAMLNet(nn.Module):
    """
    基于 LSTM 的 MAML 网络 (适用于序列数据)
    """

    def __init__(
            self,
            input_dim: int = 1,
            hidden_dim: int = 64,
            num_layers: int = 2,
            num_classes: int = 5,
            bidirectional: bool = True
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.2 if num_layers > 1 else 0
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        x: [batch, seq_length, input_dim] 或 [batch, seq_length]
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch, seq_length, 1]

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最后一个时间步的输出
        features = lstm_out[:, -1, :]  # [batch, hidden_dim * 2]

        # 分类
        logits = self.classifier(features)

        return logits
