__author__ = 'Shilong Bao'
__copyright__ = ''
__email__ = 'baoshilong@iie.ac.cn'

from typing import Optional, List, Dict, Tuple, Callable
import torch.nn as nn
import torch.nn.functional as F
import torch


class CESourceOnly(nn.Module):
    '''
    only compute empirical loss on source domain
    '''

    def __init__(self, num_classes=None, multi_label=False, **kwargs):
        super(CESourceOnly, self).__init__()
        self.multi_label = multi_label

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.multi_label:
            return F.binary_cross_entropy_with_logits(preds, targets.float())
        else:
            return F.cross_entropy(preds, targets)
