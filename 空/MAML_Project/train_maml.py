

import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
from datetime import datetime

from maml_core import MAML
from models import TimeSeriesMAMLNet, EEGMAMLNet, LSTMMAMLNet
from data_loader import SyntheticTimeSeriesDataset, SyntheticEEGDataset


def parse_args():
    parser = argparse.ArgumentParser(description='MAML Training')

    # 数据参数
    parser.add_argument('--dataset', type=str, default='timeseries',
                        choices=['timeseries', 'eeg'],
                        help='数据集类型')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='总类别数')
    parser.add_argument('--samples_per_class', type=int, default=100,
                        help='每个类别的样本数')
    parser.add_argument('--seq_length', type=int, default=128,
                        help='序列长度')

    # Few-shot 参数
    parser.add_argument('--n_way', type=int, default=5,
                        help='N-way (每个任务的类别数)')
    parser.add_argument('--k_shot', type=int, default=5,
                        help='K-shot (每个类别的支持集样本数)')
    parser.add_argument('--q_query', type=int, default=15,
                        help='每个类别的查询集样本数')

    # MAML 参数
    parser.add_argument('--inner_lr', type=float, default=0.01,
                        help='内层学习率')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='外层学习率')
    parser.add_argument('--inner_steps', type=int, default=5,
                        help='内层更新步数')
    parser.add_argument('--first_order', action='store_true',
                        help='使用一阶近似 (FOMAML)')

    # 训练参数
    parser.add_argument('--num_iterations', type=int, default=10000,
                        help='训练迭代次数')
    parser.add_argument('--meta_batch_size', type=int, default=4,
                        help='元批次大小 (每次迭代的任务数)')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='评估间隔')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='保存间隔')

    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')

    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def create_dataset(args):
    """创建数据集"""
    if args.dataset == 'timeseries':
        dataset = SyntheticTimeSeriesDataset(
            num_classes=args.num_classes,
            samples_per_class=args.samples_per_class,
            seq_length=args.seq_length
        )
        model = TimeSeriesMAMLNet(
            input_channels=1,
            seq_length=args.seq_length,
            num_classes=args.n_way
        )
    else:  # eeg
        dataset = SyntheticEEGDataset(
            num_classes=args.num_classes,
            samples_per_class=args.samples_per_class,
            num_channels=32,
            seq_length=args.seq_length
        )
        model = EEGMAMLNet(
            num_channels=32,
            seq_length=args.seq_length,
            num_classes=args.n_way
        )

    return dataset, model


def train(args):
    """主训练函数"""
    print("\n" + "=" * 70)
    print("🚀 MAML 训练开始")
    print("=" * 70)

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📌 设备: {device}")

    # 设置随机种子
    set_seed(args.seed)

    # 创建数据集和模型
    dataset, model = create_dataset(args)

    # 获取任务采样器
    task_sampler = dataset.get_task_sampler(
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query
    )

    # 创建 MAML
    maml = MAML(
        model=model,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        inner_steps=args.inner_steps,
        first_order=args.first_order,
        device=device
    )

    # 打印配置
    print(f"\n📋 训练配置:")
    print(f"   数据集: {args.dataset}")
    print(f"   N-way: {args.n_way}, K-shot: {args.k_shot}, Q-query: {args.q_query}")
    print(f"   内层学习率: {args.inner_lr}")
    print(f"   外层学习率: {args.outer_lr}")
    print(f"   内层步数: {args.inner_steps}")
    print(f"   一阶近似: {args.first_order}")
    print(f"   元批次大小: {args.meta_batch_size}")
    print(f"   总迭代次数: {args.num_iterations}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练记录
    train_losses = []
    train_accs = []
    best_acc = 0.0

    # 训练循环
    print(f"\n📈 开始训练...")
    print("-" * 70)

    pbar = tqdm(range(args.num_iterations), desc="Training")

    for iteration in pbar:
        # 采样任务批次
        task_batch = task_sampler.sample_batch(args.meta_batch_size)

        # 元训练步骤
        loss, acc = maml.meta_train_step(task_batch)

        train_losses.append(loss)
        train_accs.append(acc)

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'acc': f'{acc * 100:.2f}%'
        })

        # 定期评估
        if (iteration + 1) % args.eval_interval == 0:
            eval_accs = []
            for _ in range(100):
                task = task_sampler.sample_task()
                acc = maml.meta_test(task, inner_steps=10)
                eval_accs.append(acc)

            mean_acc = np.mean(eval_accs)
            std_acc = np.std(eval_accs)

            print(f"\n📊 Iteration {iteration + 1}:")
            print(f"   Train Loss: {np.mean(train_losses[-100:]):.4f}")
            print(f"   Train Acc: {np.mean(train_accs[-100:]) * 100:.2f}%")
            print(f"   Eval Acc: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")

            # 保存最佳模型
            if mean_acc > best_acc:
                best_acc = mean_acc
                maml.save(os.path.join(args.save_dir, 'best_model.pt'))
                print(f"   🏆 新的最佳模型! Acc: {best_acc * 100:.2f}%")

        # 定期保存
        if (iteration + 1) % args.save_interval == 0:
            maml.save(os.path.join(args.save_dir, f'model_iter_{iteration + 1}.pt'))

    # 保存最终模型
    maml.save(os.path.join(args.save_dir, 'final_model.pt'))

    # 最终评估
    print("\n" + "=" * 70)
    print("🎯 最终评估")
    print("=" * 70)

    final_accs = []
    for _ in tqdm(range(600), desc="Final Evaluation"):
        task = task_sampler.sample_task()
        acc = maml.meta_test(task, inner_steps=10)
        final_accs.append(acc)

    print(f"\n📊 最终结果:")
    print(
        f"   {args.n_way}-way {args.k_shot}-shot 准确率: {np.mean(final_accs) * 100:.2f}% ± {np.std(final_accs) * 100:.2f}%")
    print(f"   最佳验证准确率: {best_acc * 100:.2f}%")

    # 保存训练记录
    np.save(os.path.join(args.save_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(args.save_dir, 'train_accs.npy'), train_accs)

    print(f"\n✅ 训练完成！模型已保存到: {args.save_dir}")


if __name__ == '__main__':
    args = parse_args()
    train(args)
