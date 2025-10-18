import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable

import torch.nn.init as init
from torch.nn.utils.weight_norm import weight_norm

def dynamic_shuffle(q, shuffle_ratio):
    shuffled_q = q.clone()
    batch_size = q.size(0)
    seq_len = q.size(1)
    for i in range(batch_size):
        # 计算当前样本需要 shuffle 的元素数量
        num_shuffle = int(seq_len * shuffle_ratio[i])

        # 生成需要 shuffle 的随机索引
        indices = torch.randperm(seq_len)[:num_shuffle]

        # 生成不需要 shuffle 的剩余索引
        remaining_indices = torch.tensor([j for j in range(seq_len) if j not in indices])

        # 对当前样本的指定索引进行 shuffle
        shuffled_q[i, indices] = q[i, indices[torch.randperm(num_shuffle)]]

        # 保持剩余索引不变
        shuffled_q[i, remaining_indices] = q[i, remaining_indices]
    return shuffled_q

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()



class Qmodel(nn.Module):
    def __init__(self, w_emb, q_emb, q_net):
        super(Qmodel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.q_net = q_net
        # q_only_branch
        self.q_only = FCNet([self.q_emb.num_hid, 2048, 2048])
        self.q_cls = weight_norm(nn.Linear(2048, 2274), dim=None)

    def forward(self, q, ratio=None):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        return: logits
        """
        if ratio is not None:
            seq = q.size(1)
            #shuffle_ratio = ratio
            num_shuffle = int(ratio)
            indices = torch.randperm(q.size(1))[:num_shuffle]
            remaining_indices = torch.tensor([i for i in range(seq) if i not in indices])
            shuffled_q = q.clone()
            shuffled_q[:, indices] = q[:, indices[torch.randperm(num_shuffle)]]
            shuffled_q[:, remaining_indices] = q[:, remaining_indices]

            w_emb = self.w_emb(shuffled_q)
        else:

            w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb)  # [batch, q_dim]

        q_emb_only = q_emb.detach()
        q_only_emb = self.q_only(q_emb_only)  # [batch, num_hid]
        q_only_logits = self.q_cls(q_only_emb)



        return q_only_logits



def build_Qmodel(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    q_net = FCNet([q_emb.num_hid, num_hid])


    return Qmodel(w_emb, q_emb, q_net)