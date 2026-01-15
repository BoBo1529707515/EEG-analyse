"""
MAML 安装验证测试
"""

import torch
import torch.nn as nn
from models import TimeSeriesMAMLNet, EEGMAMLNet, LSTMMAMLNet
from maml_core import MAML


def test_maml_setup():
    print("=" * 60)
    print("🔍 MAML 环境测试")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📌 使用设备: {device}")

    # 测试模型创建
    print("\n📦 测试模型创建...")

    # 1. 时间序列模型
    ts_model = TimeSeriesMAMLNet(
        input_channels=1,
        seq_length=128,
        num_classes=5
    )
    print(f"   ✅ TimeSeriesMAMLNet: {sum(p.numel() for p in ts_model.parameters())} 参数")

    # 2. EEG 模型
    eeg_model = EEGMAMLNet(
        num_channels=32,
        seq_length=256,
        num_classes=4
    )
    print(f"   ✅ EEGMAMLNet: {sum(p.numel() for p in eeg_model.parameters())} 参数")

    # 3. LSTM 模型
    lstm_model = LSTMMAMLNet(
        input_dim=1,
        hidden_dim=64,
        num_classes=5
    )
    print(f"   ✅ LSTMMAMLNet: {sum(p.numel() for p in lstm_model.parameters())} 参数")

    # 测试 MAML 初始化
    print("\n🚀 测试 MAML 初始化...")
    maml = MAML(
        model=ts_model,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5,
        device=device
    )
    print("   ✅ MAML 初始化成功!")

    # 测试前向传播
    print("\n🧪 测试前向传播...")

    # 创建模拟数据
    batch_size = 4
    seq_length = 128

    # 模拟 support set 和 query set
    support_x = torch.randn(5, seq_length).to(device)  # 5-shot
    support_y = torch.tensor([0, 1, 2, 3, 4]).to(device)  # 5-way
    query_x = torch.randn(10, seq_length).to(device)  # 10 query samples
    query_y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]).to(device)

    # 测试内层循环
    ts_model.to(device)
    params = dict(ts_model.named_parameters())
    adapted_params = maml.inner_loop(support_x, support_y, params)
    print("   ✅ 内层循环 (Inner Loop) 测试通过!")

    # 测试完整训练步骤
    print("\n📈 测试训练步骤...")
    task = (support_x, support_y, query_x, query_y)
    loss, acc = maml.meta_train_step([task])
    print(f"   ✅ Meta-train step 完成!")
    print(f"      Loss: {loss:.4f}")
    print(f"      Accuracy: {acc * 100:.2f}%")

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！MAML 环境配置成功！")
    print("=" * 60)


if __name__ == "__main__":
    test_maml_setup()
