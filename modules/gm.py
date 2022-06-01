from typing import Optional, List, Dict, Tuple, Callable
import torch.nn as nn
import torch.nn.functional as F
import torch

from .grl import WarmStartGradientReverseLayer

class GeneralModule(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: nn.Module,
                 head: nn.Module, adv_head: nn.Module, grl: Optional[WarmStartGradientReverseLayer] = None,
                 finetune: Optional[bool] = True):
        super(GeneralModule, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.bottleneck = bottleneck
        self.head = head
        self.adv_head = adv_head
        self.finetune = finetune
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=500,
                                                       auto_step=False) if grl is None else grl

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        features = self.backbone(x)
        features = self.bottleneck(features)
        outputs = self.head(features)
        if self.adv_head is not None:
            features_adv = self.grl_layer(features)
            outputs_adv = self.adv_head(features_adv)
        else:
            outputs_adv = outputs
        if self.training:
            return outputs, outputs_adv
        else:
            return outputs

    def step(self):
        """
        Gradually increase :math:`\lambda` in GRL layer.
        """
        self.grl_layer.step()

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """
        Return a parameters list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer.
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else base_lr},
            {"params": self.bottleneck.parameters(), "lr": base_lr},
            {"params": self.head.parameters(), "lr": base_lr},
            # {"params": self.adv_head.parameters(), "lr": base_lr}
        ]
        if self.adv_head is not None:
            params += [{"params": self.adv_head.parameters(), "lr": base_lr}]
        return params

    def get_classification_parameters(self, base_lr = 1.0) -> List[Dict]:

        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else base_lr},
            {"params": self.bottleneck.parameters(), "lr": base_lr},
            {"params": self.head.parameters(), "lr": base_lr}
        ]
        return params
    
    def get_adversial_parameters(self, base_lr = 1.0) -> List[Dict]:
        
        if self.adv_head is not None:
            params = [{"params": self.adv_head.parameters(), "lr": base_lr}]
            return params
        else:
            assert False

class ImageClassifier(GeneralModule):
    r"""Classifier for MDD.
    Classifier for MDD has one backbone, one bottleneck, while two classifier heads.
    The first classifier head is used for final predictions.
    The adversarial classifier head is only used when calculating MarginDisparityDiscrepancy.
    Args:
        backbone (torch.nn.Module): Any backbone to extract 1-d features from data
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1024
        width (int, optional): Feature dimension of the classifier head. Default: 1024
        grl (nn.Module): Gradient reverse layer. Will use default parameters if None. Default: None.
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
    Inputs:
        - x (tensor): input data
    Outputs:
        - outputs: logits outputs by the main classifier
        - outputs_adv: logits outputs by the adversarial classifier
    Shape:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, C)`, where C means the number of classes.
    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,
            >>> # x is inputs, classifier is an ImageClassifier
            >>> outputs, outputs_adv = classifier(x)
            >>> classifier.step()
    """

    def __init__(self, backbone: nn.Module, num_classes: int,
                 bottleneck_dim: Optional[int] = 1024, width: Optional[int] = 1024,
                 grl: Optional[WarmStartGradientReverseLayer] = None, finetune=True, pool_layer=None,
                 require_adv=True):
        grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=500,
                                                       auto_step=False) if grl is None else grl

        # if pool_layer is None:
        #     pool_layer = nn.Sequential(
        #         nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        #         nn.Flatten()
        #     )
        bottleneck = nn.Sequential(
            # pool_layer,
            # nn.Linear(2048, bottleneck_dim),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        bottleneck[1].weight.data.normal_(0, 0.005)
        bottleneck[1].bias.data.fill_(0.1)

        # The classifier head used for final predictions.
        head = nn.Sequential(
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, num_classes)
        )
        # The adversarial classifier head
        if require_adv:
            adv_head = nn.Sequential(
                nn.Linear(bottleneck_dim, width),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(width, num_classes)
            )
        else:
            adv_head = None
        for dep in range(2):
            head[dep * 3].weight.data.normal_(0, 0.01)
            head[dep * 3].bias.data.fill_(0.0)
            if require_adv:
                adv_head[dep * 3].weight.data.normal_(0, 0.01)
                adv_head[dep * 3].bias.data.fill_(0.0)
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck,
                                              head, adv_head, grl_layer, finetune)