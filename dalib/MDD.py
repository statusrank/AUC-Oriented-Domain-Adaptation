from typing import Optional, List, Dict, Tuple, Callable
import torch.nn as nn
import torch.nn.functional as F
import torch

class MarginDisparityDiscrepancy(nn.Module):
    r"""The margin disparity discrepancy (MDD) proposed in `Bridging Theory and Algorithm for Domain Adaptation (ICML 2019) <https://arxiv.org/abs/1904.05801>`_.
    MDD can measure the distribution discrepancy in domain adaptation.
    The :math:`y^s` and :math:`y^t` are logits output by the main head on the source and target domain respectively.
    The :math:`y_{adv}^s` and :math:`y_{adv}^t` are logits output by the adversarial head.
    The definition can be described as:
    .. math::
        \mathcal{D}_{\gamma}(\hat{\mathcal{S}}, \hat{\mathcal{T}}) =
        -\gamma \mathbb{E}_{y^s, y_{adv}^s \sim\hat{\mathcal{S}}} L_s (y^s, y_{adv}^s) +
        \mathbb{E}_{y^t, y_{adv}^t \sim\hat{\mathcal{T}}} L_t (y^t, y_{adv}^t),
    where :math:`\gamma` is a margin hyper-parameter, :math:`L_s` refers to the disparity function defined on the source domain
    and :math:`L_t` refers to the disparity function defined on the target domain.
    Args:
        source_disparity (callable): The disparity function defined on the source domain, :math:`L_s`.
        target_disparity (callable): The disparity function defined on the target domain, :math:`L_t`.
        margin (float): margin :math:`\gamma`. Default: 4
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    Inputs:
        - y_s: output :math:`y^s` by the main head on the source domain
        - y_s_adv: output :math:`y^s` by the adversarial head on the source domain
        - y_t: output :math:`y^t` by the main head on the target domain
        - y_t_adv: output :math:`y_{adv}^t` by the adversarial head on the target domain
        - w_s (optional): instance weights for source domain
        - w_t (optional): instance weights for target domain
    Examples::
        >>> num_outputs = 2
        >>> batch_size = 10
        >>> loss = MarginDisparityDiscrepancy(margin=4., source_disparity=F.l1_loss, target_disparity=F.l1_loss)
        >>> # output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_outputs), torch.randn(batch_size, num_outputs)
        >>> # adversarial output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_outputs), torch.randn(batch_size, num_outputs)
        >>> output = loss(y_s, y_s_adv, y_t, y_t_adv)
    """

    def __init__(self, source_disparity: Callable, target_disparity: Callable,
                 margin: Optional[float] = 4, reduction: Optional[str] = 'mean', **kwargs):
        super(MarginDisparityDiscrepancy, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.source_disparity = source_disparity
        self.target_disparity = target_disparity

    def forward(self, y_s: torch.Tensor, y_s_adv: torch.Tensor, y_t: torch.Tensor, y_t_adv: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:

        source_loss = -self.margin * self.source_disparity(y_s, y_s_adv)
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


class ClassificationMarginDisparityDiscrepancy(MarginDisparityDiscrepancy):
    r"""
    The margin disparity discrepancy (MDD) proposed in `Bridging Theory and Algorithm for Domain Adaptation (ICML 2019) <https://arxiv.org/abs/1904.05801>`_.
    It measures the distribution discrepancy in domain adaptation
    for classification.
    When margin is equal to 1, it's also called disparity discrepancy (DD).
    The :math:`y^s` and :math:`y^t` are logits output by the main classifier on the source and target domain respectively.
    The :math:`y_{adv}^s` and :math:`y_{adv}^t` are logits output by the adversarial classifier.
    They are expected to contain raw, unnormalized scores for each class.
    The definition can be described as:
    .. math::
        \mathcal{D}_{\gamma}(\hat{\mathcal{S}}, \hat{\mathcal{T}}) =
        \gamma \mathbb{E}_{y^s, y_{adv}^s \sim\hat{\mathcal{S}}} \log\left(\frac{\exp(y_{adv}^s[h_{y^s}])}{\sum_j \exp(y_{adv}^s[j])}\right) +
        \mathbb{E}_{y^t, y_{adv}^t \sim\hat{\mathcal{T}}} \log\left(1-\frac{\exp(y_{adv}^t[h_{y^t}])}{\sum_j \exp(y_{adv}^t[j])}\right),
    where :math:`\gamma` is a margin hyper-parameter and :math:`h_y` refers to the predicted label when the logits output is :math:`y`.
    You can see more details in `Bridging Theory and Algorithm for Domain Adaptation <https://arxiv.org/abs/1904.05801>`_.
    Args:
        margin (float): margin :math:`\gamma`. Default: 4
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    Inputs:
        - y_s: logits output :math:`y^s` by the main classifier on the source domain
        - y_s_adv: logits output :math:`y^s` by the adversarial classifier on the source domain
        - y_t: logits output :math:`y^t` by the main classifier on the target domain
        - y_t_adv: logits output :math:`y_{adv}^t` by the adversarial classifier on the target domain
    Shape:
        - Inputs: :math:`(minibatch, C)` where C = number of classes, or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
          with :math:`K \geq 1` in the case of `K`-dimensional loss.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then the same size as the target: :math:`(minibatch)`, or
          :math:`(minibatch, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of K-dimensional loss.
    Examples::
        >>> num_classes = 2
        >>> batch_size = 10
        >>> loss = ClassificationMarginDisparityDiscrepancy(margin=4.)
        >>> # logits output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> # adversarial logits output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> output = loss(y_s, y_s_adv, y_t, y_t_adv)
    """

    def __init__(self, margin: Optional[float] = 4, **kwargs):
        def source_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            _, prediction = y.max(dim=1)
            return F.cross_entropy(y_adv, prediction, reduction='none')

        def target_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            _, prediction = y.max(dim=1)
            return -F.nll_loss(shift_log(1. - F.softmax(y_adv, dim=1)), prediction, reduction='none')

        super(ClassificationMarginDisparityDiscrepancy, self).__init__(source_discrepancy, target_discrepancy, margin, **kwargs)


class ClassificationMarginDisparityDiscrepancyMultiLabel(MarginDisparityDiscrepancy):
    def __init__(self, margin: Optional[float] = 4, **kwargs):
        def source_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            # _, prediction = y.max(dim=1)
            prediction = (y > 0).float().view(-1)
            y_adv = y_adv.view(-1)
            return F.binary_cross_entropy_with_logits(y_adv, prediction, reduction='none')

        def target_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            # _, prediction = y.max(dim=1)
            prediction = (y > 0).long().view(-1)
            # F.softmax(y_adv, dim=1)
            y_adv_sigmoid = F.sigmoid(y_adv).view(-1, 1)
            return -F.nll_loss(shift_log(1. - torch.cat([1 - y_adv_sigmoid, y_adv_sigmoid], dim=1)), prediction, reduction='none')

        super(ClassificationMarginDisparityDiscrepancyMultiLabel, self).__init__(source_discrepancy, target_discrepancy, margin, **kwargs)


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
