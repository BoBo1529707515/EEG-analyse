"""
MAML (Model-Agnostic Meta-Learning) 核心实现
适用于时间序列/EEG分析任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from copy import deepcopy


class MAML:
    """
    MAML 元学习算法

    参数:
        model: 基础模型
        inner_lr: 内层更新学习率 (任务适应)
        outer_lr: 外层更新学习率 (元学习)
        inner_steps: 内层更新步数
        first_order: 是否使用一阶近似 (FOMAML)
    """

    def __init__(
            self,
            model: nn.Module,
            inner_lr: float = 0.01,
            outer_lr: float = 0.001,
            inner_steps: int = 5,
            first_order: bool = False,
            device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.device = device

        # 外层优化器
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=outer_lr
        )

    def inner_loop(self, support_x, support_y, params=None):
        """
        内层循环：在单个任务上进行快速适应

        参数:
            support_x: 支持集输入 [N_way * K_shot, ...]
            support_y: 支持集标签 [N_way * K_shot]
            params: 当前模型参数 (如果为None则使用模型参数)

        返回:
            adapted_params: 适应后的参数
        """
        if params is None:
            params = dict(self.model.named_parameters())

        # 内层更新
        for step in range(self.inner_steps):
            # 前向传播
            logits = self.functional_forward(support_x, params)
            loss = F.cross_entropy(logits, support_y)

            # 计算梯度
            grads = grad(
                loss,
                params.values(),
                create_graph=not self.first_order,
                retain_graph=not self.first_order,
                allow_unused=True
            )

            # 更新参数
            params = {
                name: param - self.inner_lr * g if g is not None else param
                for (name, param), g in zip(params.items(), grads)
            }

        return params

    def functional_forward(self, x, params):
        """
        使用给定参数进行前向传播（函数式）
        """
        # 临时替换模型参数
        original_params = {}
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
            param.data = params[name]

        # 前向传播
        output = self.model(x)

        # 恢复原始参数
        for name, param in self.model.named_parameters():
            param.data = original_params[name]

        return output

    def meta_train_step(self, task_batch):
        """
        元训练步骤

        参数:
            task_batch: 任务批次列表
                每个任务包含 (support_x, support_y, query_x, query_y)

        返回:
            meta_loss: 元损失
            accuracies: 各任务准确率
        """
        self.model.train()
        meta_loss = 0.0
        accuracies = []

        for task in task_batch:
            support_x, support_y, query_x, query_y = task

            # 移动到设备
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)

            # 获取当前参数
            params = dict(self.model.named_parameters())

            # 内层适应
            adapted_params = self.inner_loop(support_x, support_y, params)

            # 在查询集上计算损失
            query_logits = self.functional_forward(query_x, adapted_params)
            task_loss = F.cross_entropy(query_logits, query_y)
            meta_loss += task_loss

            # 计算准确率
            with torch.no_grad():
                pred = query_logits.argmax(dim=1)
                acc = (pred == query_y).float().mean().item()
                accuracies.append(acc)

        # 外层更新
        meta_loss = meta_loss / len(task_batch)

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item(), sum(accuracies) / len(accuracies)

    def meta_test(self, task, inner_steps=None):
        """
        元测试：在新任务上进行适应和评估
        """
        self.model.eval()

        if inner_steps is None:
            inner_steps = self.inner_steps

        support_x, support_y, query_x, query_y = task

        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)
        query_x = query_x.to(self.device)
        query_y = query_y.to(self.device)

        # 适应
        params = dict(self.model.named_parameters())

        # 临时增加内层步数进行更充分的适应
        original_steps = self.inner_steps
        self.inner_steps = inner_steps

        with torch.no_grad():
            # 使用 FOMAML 模式进行测试
            for step in range(inner_steps):
                logits = self.model(support_x)
                loss = F.cross_entropy(logits, support_y)
                grads = torch.autograd.grad(loss, self.model.parameters())

                for param, g in zip(self.model.parameters(), grads):
                    param.data = param.data - self.inner_lr * g

        # 评估
        with torch.no_grad():
            query_logits = self.model(query_x)
            pred = query_logits.argmax(dim=1)
            accuracy = (pred == query_y).float().mean().item()

        self.inner_steps = original_steps

        return accuracy

    def save(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'inner_lr': self.inner_lr,
            'inner_steps': self.inner_steps,
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.inner_lr = checkpoint['inner_lr']
        self.inner_steps = checkpoint['inner_steps']
