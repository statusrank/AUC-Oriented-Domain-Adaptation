__author__ = 'Shilong Bao'
__copyright__ = ''
__email__ = 'baoshilong@iie.ac.cn'

from typing import Optional, List, Dict, Tuple, Callable
import torch.nn as nn
import torch.nn.functional as F
import torch
from libauc.losses import AUCMLoss, AUCM_MultiLabel

from modules.grl import GradientReverseLayer, WarmStartGradientReverseLayer

# def l_epsilon(x: torch.Tensor, gamma: float, theta: float, epsilon: float = 1e-6) -> torch.Tensor:

#     return F.relu((x - epsilon) / (gamma - theta)) + F.relu((-x - epsilon) / (gamma - theta))

def get_loss_function(loss_type: str):

    if loss_type == 'log_loss':
        return lambda x, epsilon: \
                torch.log(1 + torch.exp(- (x - epsilon))) + torch.log(1 + torch.exp(x + epsilon))
    elif loss_type == 'max_loss':
        return lambda x, epsilon: \
                F.relu(x - epsilon) + F.relu(-x - epsilon)
    else:
        raise KeyError("Not support: {}".format(loss_type))

class AUCMSourceOnly(nn.Module):
    '''
    The loss function could be regarded as an ablation study about AUC without domain adaption. 
    '''

    def __init__(self, num_classes,
                imratio,
                multi_label = False, 
                **kwargs):
        
        super(AUCMSourceOnly, self).__init__()
        
        # print("====> imratio:", imratio)

        if multi_label:
            self.loss_func = AUCM_MultiLabel(imratio=imratio, num_classes=num_classes)
        else:
            self.loss_func = AUCMLoss(imratio)
        self.num_classes = num_classes
        self.multi_label = multi_label

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        
        def CalLossPerClass(pred, target):

            return self.loss_func(pred, target)

        if not self.multi_label:
            preds = F.softmax(preds, dim = -1)
            Y = torch.stack([
                targets.eq(i).float() for i in range(self.num_classes)], 1).squeeze()
            return CalLossPerClass(preds[:, 1:], Y[:, 1:])
        else:
            preds = F.sigmoid(preds)
            Y = targets
            return CalLossPerClass(preds, Y)