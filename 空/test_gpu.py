"""
GPU 环境测试脚本
在 PyCharm 中运行此文件验证配置是否成功
"""

import torch
import sys

print("=" * 50)
print("🔍 环境检测报告")
print("=" * 50)

print(f"\n📌 Python 版本: {sys.version}")
print(f"📌 Python 路径: {sys.executable}")
print(f"📌 PyTorch 版本: {torch.__version__}")

print(f"\n🎮 CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   ├─ CUDA 版本: {torch.version.cuda}")
    print(f"   ├─ GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"   ├─ GPU 数量: {torch.cuda.device_count()}")

    # 获取显存信息
    props = torch.cuda.get_device_properties(0)
    total_mem = props.total_memory / 1e9
    print(f"   └─ 显存大小: {total_mem:.2f} GB")

    # GPU 计算测试
    print("\n🧪 GPU 计算测试...")
    try:
        x = torch.randn(2000, 2000, device='cuda')
        y = torch.matmul(x, x)
        torch.cuda.synchronize()  # 等待 GPU 计算完成
        print("   ✅ GPU 矩阵乘法测试通过！")

        # 显示当前显存使用
        allocated = torch.cuda.memory_allocated(0) / 1e6
        print(f"   📊 当前显存占用: {allocated:.1f} MB")
    except Exception as e:
        print(f"   ❌ GPU 测试失败: {e}")
else:
    print("\n⚠️  CUDA 不可用！请检查：")
    print("   1. 是否安装了 NVIDIA 驱动")
    print("   2. 是否安装了带 CUDA 支持的 PyTorch")
    print("   3. PyCharm 是否选择了正确的 Conda 环境")

print("\n" + "=" * 50)
print("检测完成！")
print("=" * 50)
