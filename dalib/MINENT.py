import torch.nn as nn
import torch.nn.functional as F
import torch
from math import log


class EntropyMinimization(nn.Module):
    def __init__(self, eps=1e-6, multi_label=False, **kwargs):
        super(EntropyMinimization, self).__init__()
        self.multi_label = multi_label
        self.eps = eps

    def forward(self, y_s, y_s_adv, y_t, y_t_adv):
        if self.multi_label:
            p = torch.sigmoid(y_t)
        else:
            p = torch.softmax(y_t, dim=1)
        neg_ent = p * torch.log(p + self.eps)
        neg_ent = neg_ent.sum(-1) / log(p.shape[1])
        return neg_ent.mean()
