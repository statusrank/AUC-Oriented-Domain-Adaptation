from typing import Optional, List, Dict, Tuple, Callable
import torch.nn as nn
import torch.nn.functional as F
import torch


def MAE_Loss(pred: torch.Tensor, target: torch.Tensor):

    target_to_one_hot = torch.zeros_like(pred).scatter_(1, target.view(-1, 1), 1.)
    return F.l1_loss(F.softmax(pred, dim = -1), target_to_one_hot, reduction='mean')

class F_DAL(nn.Module):

    def __init__(self, 
                gamma: Optional[int] = 1.0,
                divergence_type='Pearson',
                reduction: Optional[str] = 'mean', 
                **kwargs):
        super(F_DAL, self).__init__()

        self.divergence_type = divergence_type
        self.reduction = reduction

        self.gamma = gamma
        self.get_divergence()

    def forward(self, y_s: torch.Tensor, y_s_adv: torch.Tensor, y_t: torch.Tensor, y_t_adv: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        source_loss = self.source_disparity(y_s, y_s_adv)
        target_loss = self.target_disparity(y_t, y_t_adv)
        if w_s is None:
            w_s = torch.ones_like(source_loss)
        source_loss = source_loss * w_s
        if w_t is None:
            w_t = torch.ones_like(target_loss)
        target_loss = target_loss * w_t

        loss = source_loss + target_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

    def pre_processing_for_discrepancy(self, y: torch.Tensor, y_adv: torch.Tensor):
        
        _, arg_index = y.max(dim = 1)
        y_adv_argmax_y = torch.index_select(y_adv, dim=-1, index = arg_index)

        return y_adv_argmax_y

    def get_divergence(self):
        
        def source_discrepancy_Pearson(y: torch.Tensor, y_adv: torch.Tensor):

            return self.pre_processing_for_discrepancy(y, y_adv)

        def target_discrepancy_Pearson(y: torch.Tensor, y_adv: torch.Tensor):

            y_adv_argmax_y = self.pre_processing_for_discrepancy(y, y_adv)
            return - (y_adv_argmax_y.square() / 4 + y_adv_argmax_y)
        
        def source_discrepancy_gamma_JS(y: torch.Tensor, y_adv: torch.Tensor):

            y_adv_argmax_y = self.pre_processing_for_discrepancy(y, y_adv)
            return shift_log(2.0 / (1 + torch.exp(-y_adv_argmax_y)))

        def target_discrepancy_gamma_JS(y: torch.Tensor, y_adv: torch.Tensor):
            y_adv_argmax_y = self.pre_processing_for_discrepancy(y, y_adv)
            log_loss = shift_log(2.0 / (1 + torch.exp(-y_adv_argmax_y)))
            return shift_log(2 - torch.exp(log_loss))

        
        assert self.divergence_type in ['Pearson', 'gamma-JS']

        if self.divergence_type == 'Pearson':
            self.source_disparity = source_discrepancy_Pearson
            self.target_disparity = target_discrepancy_Pearson
        elif self.divergence_type == 'gamma-JS':
            self.source_disparity = source_discrepancy_gamma_JS
            self.target_disparity = target_discrepancy_gamma_JS
        else:
            raise TypeError('Not Implementation Error')

def shift_log(x: torch.Tensor, offset: Optional[float] = 1e-6) -> torch.Tensor:
    '''
       First shift, then calculate log, which can be described as
            y = \max(log(x+\text{offset}), 0)
        Used to avoid the gradient explosion problem in log(x) function when x=0.
        Args:
            x (torch.Tensor): input tensor
            offset (float, optional): offset size. Default: 1e-6
        .. note::
            Input tensor falls in [0., 1.] and the output tensor falls in [-log(offset), 0]
    '''
    return torch.log(torch.clamp(x + offset, max=1.))