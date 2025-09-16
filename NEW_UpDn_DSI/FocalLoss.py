# FocalLoss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma


    def forward(self, inputs, labels):
        # 对模型输出进行sigmoid激活，得到预测概率
        probs = torch.sigmoid(inputs)

        # 限制概率值以避免数值问题
        probs = probs.clamp(min=0.0001, max=1.0)

        # 计算焦点损失的公式
        loss = -self.alpha * (1 - probs).pow(self.gamma) * labels * torch.log(probs)

        return loss
