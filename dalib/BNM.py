import torch.nn as nn
import torch.nn.functional as F
import torch


class BatchNuclearnormMaximization(nn.Module):
    def __init__(self, multi_label=False, **kwargs):

        self.multi_label = multi_label
        super(BatchNuclearnormMaximization, self).__init__()

    def forward(self, y_s, y_s_adv, y_t, y_t_adv):
        if not self.multi_label:
            y_t = torch.softmax(y_t, dim=1)
        else:
            y_t = torch.sigmoid(y_t)
        return torch.mean(torch.svd(y_t)[1])