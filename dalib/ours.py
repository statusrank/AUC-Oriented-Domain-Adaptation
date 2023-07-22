__author__ = 'Shilong Bao'
__copyright__ = ''
__email__ = 'baoshilong@iie.ac.cn'

from typing import Optional, List, Dict, Tuple, Callable
import torch.nn as nn
import torch.nn.functional as F
import torch
from modules.grl import GradientReverseLayer

def get_loss_function(loss_type: str):

    if loss_type == 'log_loss':
        return lambda x, epsilon: \
                torch.log(1 + torch.exp(- (x - epsilon))) + torch.log(1 + torch.exp(x + epsilon))
    elif loss_type == 'max_loss':
        return lambda x, epsilon: \
                F.relu(x - epsilon) + F.relu(-x - epsilon)
    else:
        raise KeyError("Not support: {}".format(loss_type))

class AUCSourceOnly(nn.Module):
    '''
    The loss function could be regarded as an ablation study about AUC without domain adaption. 
    '''

    def __init__(self, num_classes,
                epsilon = 0.05, 
                loss_type = 'log_loss',
                multi_label = False, 
                **kwargs):
        
        super(AUCSourceOnly, self).__init__()
        
        self.epsilon = epsilon
        self.loss_func = get_loss_function(loss_type)

        self.num_classes = num_classes

        self.multi_label = multi_label

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:


        # print(preds.shape)

        if not self.multi_label:
            preds = F.softmax(preds, dim = -1)
            Y = torch.stack([
                targets.eq(i).float() for i in range(self.num_classes)], 1).squeeze()
        else:
            preds = F.sigmoid(preds)
            Y = targets
        # print(Y.shape)
        # print('Y: ', Y)

        N_per_class = Y.sum(0)

        # print('N_per_class: ', N_per_class.sum())
        # print("target.shape: ", targets.unique().shape[0])
        def CalLossPerClass(pred, target, fac):
            
            # print(target.shape)
            # print('target: ', target)

            pred_poss = torch.index_select(pred, 0, target.nonzero().view(-1))
            pred_negs = torch.index_select(pred, 0, (1 - target).nonzero().view(-1))

            delta_f_pos_negs = (pred_poss.view(-1, 1) - pred_negs.view(1, -1))

            # print("delta_f_pos_negs: ", delta_f_pos_negs)
            x = 4 * (1.0 - delta_f_pos_negs)

            return self.loss_func(x, self.epsilon).sum() * fac
        
        loss = torch.tensor([0.0]).cuda()

        # print('N_per_class: ', N_per_class)

        for i in range(self.num_classes):
            # if N_per_class[i] == 0:
            #     continue
            fac = 1.0 / (N_per_class[i] * (N_per_class.sum() - N_per_class[i]))

            Yi, predi = Y[:, i], preds[:, i]

            # print(Yi)
            # print(predi)

            loss += CalLossPerClass(predi, Yi, fac)

        return loss

class AUCDomainAdapation(nn.Module):
    '''

    '''
    def __init__(self, 
                num_classes, 
                probs,
                beta1=1.0,
                beta2=1.0,
                epsilon = 0.05,
                gamma = 1.0,
                warm_up_epoch = 10,
                loss_type='log_loss',
                multi_label = False, 
                use_pseudo_label = True,
                class_thres = 0.5,
                grl: Optional[bool] = None,
                **kwargs):
        super(AUCDomainAdapation, self).__init__()
        
        self.probs = probs 
        self.epsilon = epsilon

        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma

        self.num_classes = num_classes

        self.loss_func = get_loss_function(loss_type)

        self.grl_layer = GradientReverseLayer() if grl is None else grl 

        self.warm_up_epoch = warm_up_epoch

        self.multi_label = multi_label

        self.use_pseudo_label = use_pseudo_label

        self.class_thres = class_thres

    def empirical_error(self, preds: torch.Tensor, targets: torch.Tensor):

        def CalLossPerClass_error(pred, target):
            
            # source empirical
            pred_poss = torch.index_select(pred, 0, target.nonzero().view(-1)) # (num_pos, 1)
            pred_negs = torch.index_select(pred, 0, (1 - target).nonzero().view(-1)) # (num_negs, 1)
                        
            delta_f_pos_negs = (pred_poss.view(-1, 1) - pred_negs.view(1, -1))
            empirical_x = 4 * (1 - delta_f_pos_negs)  # (num_poss, num_negs)

            return self.loss_func(empirical_x, self.epsilon).sum()

        if not self.multi_label:
            Y = torch.stack([
                targets.eq(i).float() for i in range(self.num_classes)], 1).squeeze()
            preds = F.softmax(preds, dim = -1)
        else:
            Y = targets   
            preds = F.sigmoid(preds) 
        
        N_per_class = Y.sum(0) # (num_classes, 1)

        empirical_loss = torch.tensor([0.0]).cuda()

        for i in range(self.num_classes):
            fac = 1.0 / (N_per_class[i] * (N_per_class.sum() - N_per_class[i]))

            Yi, predi = Y[:, i], preds[:, i]
            batch_empirical_loss = CalLossPerClass_error(predi, Yi) * fac
            
            empirical_loss += batch_empirical_loss
        return empirical_loss

    def source_discrepency(self, preds: torch.Tensor, targets: torch.Tensor, pred_advs: torch.Tensor):
        '''
        Note:
            To optimize AUC, we rewrite the sampler mechanism in dataloader to ensure each class at least haveing an example 
        in every mini-batch. 
        '''

        def CalLossPerClass_discrepancy(pred, pred_adv, target):
            
            # source empirical
            pred_poss = torch.index_select(pred, 0, target.nonzero().view(-1)) # (num_pos, 1)
            pred_negs = torch.index_select(pred, 0, (1 - target).nonzero().view(-1)) # (num_negs, 1)
                        
            delta_f_pos_negs = pred_poss.view(-1, 1) - pred_negs.view(1, -1)

            # source discrepancy
            pred_adv_pos = torch.index_select(pred_adv, 0, target.nonzero().view(-1))
            pred_adv_neg = torch.index_select(pred_adv, 0, (1 - target).nonzero().view(-1))
            
            delta_f_hat_pos_negs = pred_adv_pos.view(-1, 1) - pred_adv_neg.view(1, -1)
            source_x = 2 * (delta_f_hat_pos_negs - delta_f_pos_negs)

            return self.loss_func(source_x, self.epsilon).sum()

        if not self.multi_label:
            Y = torch.stack([
                targets.eq(i).float() for i in range(self.num_classes)], 1).squeeze()
            
            preds = F.softmax(preds, dim = -1)
            pred_advs = F.softmax(pred_advs, dim = -1)
        else:
            Y = targets

            preds = F.sigmoid(preds)
            pred_advs = F.sigmoid(pred_advs)
        
        N_per_class = Y.sum(0) # (num_classes, 1)

        discrepancy_loss = torch.tensor([0.0]).cuda()
        
        for i in range(self.num_classes):
            fac = 1.0 / (N_per_class[i] * (N_per_class.sum() - N_per_class[i]))

            Yi, predi, predi_adv = Y[:, i], preds[:, i], pred_advs[:, i]
            batch_discrepancy_loss = fac * CalLossPerClass_discrepancy(predi, predi_adv, Yi)
            
            discrepancy_loss += batch_discrepancy_loss

        return discrepancy_loss

    
    def target_discrepancy_loss(self, preds: torch.Tensor, pred_advs: torch.tensor, epoch: int = 0) -> torch.Tensor:
        
        def generate_pseudo_label(probs_to_y_hat: torch.Tensor):
            if not self.multi_label:
                pseudo_y = probs_to_y_hat.argmax(dim = -1) # (batch, C) -> (batch, )
                one_hot_pseudo_y = torch.stack([
                    pseudo_y.eq(i).float() for i in range(self.num_classes)], 1).squeeze() # (batch, ) -> (batch, C)
            else:
                one_hot_pseudo_y = probs_to_y_hat.ge(self.class_thres).float() # (batch, C)
            return one_hot_pseudo_y

        if not self.multi_label:
            preds = F.softmax(preds, dim = -1)
            pred_advs = F.softmax(pred_advs, dim = -1)
        else:
            preds = F.sigmoid(preds)
            pred_advs = F.sigmoid(pred_advs)
        
        batch_N = preds.shape[0]

        if self.use_pseudo_label:
            Y = generate_pseudo_label(pred_advs)
            N_per_class = Y.sum(0)

        def CalPerSamples(predi, predi_adv, target=None):
            
            if target is not None:
                # source empirical
                pred_poss = torch.index_select(predi, 0, target.nonzero().view(-1)) # (num_pos, 1)
                pred_negs = torch.index_select(predi, 0, (1 - target).nonzero().view(-1)) # (num_negs, 1)
                            
                delta_f = pred_poss.view(-1, 1) - pred_negs.view(1, -1)

                # source discrepancy
                pred_adv_pos = torch.index_select(predi_adv, 0, target.nonzero().view(-1))
                pred_adv_neg = torch.index_select(predi_adv, 0, (1 - target).nonzero().view(-1))
                
                delta_f_hat = pred_adv_pos.view(-1, 1) - pred_adv_neg.view(1, -1)
            else:
                delta_f = predi.view(-1, 1) - predi.view(1, -1)      

                delta_f_hat = predi_adv.view(-1, 1) - predi_adv.view(1, -1)
                
            
            target_x = 2 * (delta_f_hat - delta_f)
            
            return self.loss_func(target_x, self.epsilon).sum()

        discrepancy_loss = torch.tensor([0.0]).cuda()
        for i in range(self.num_classes):

            if self.use_pseudo_label:
                fac = 1.0 / (N_per_class[i] * (N_per_class.sum() - N_per_class[i]))
            else:
                fac = 1.0 / (self.probs[i] * (1 - self.probs[i]) * batch_N ** 2) 
            
            predi, predi_adv = preds[:, i], pred_advs[:, i]

            Yi = Y[:, i] if self.use_pseudo_label else None

            # print("=====> predi: ", predi)
            # print("=====> predi_adv:", predi_adv)

            discrepancy_loss += fac * CalPerSamples(predi, predi_adv, Yi)

            # print("=====> CalPerSamples: ", fac * CalPerSamples(predi, predi_adv))

        return discrepancy_loss

    def forward(self, y_s: torch.Tensor, y_s_adv: torch.Tensor, labels_s: torch.Tensor, 
                y_t: torch.Tensor, y_t_adv: torch.Tensor, epoch: int = 0) -> torch.Tensor:

        source_empirical_error = self.empirical_error(y_s, labels_s)

        # y_grl = self.grl_layer(torch.cat((y_s_adv, y_t_adv), dim = 0))
        # y_s_adv, y_t_adv = y_grl.chunk(2, dim = 0)

        # source_discrepancy = self.source_discrepency(y_s, labels_s, y_s_adv)

        # # target_discrepancy = self.target_discrepancy_loss(y_t, y_t_adv, epoch) 
        # transfer_loss = -self.beta2 * 0.5 * source_discrepancy
        # if epoch >= self.warm_up_epoch:
        #     target_discrepancy = self.target_discrepancy_loss(y_t, y_t_adv, epoch)  
        #     transfer_loss += self.beta1 * 0.25 * target_discrepancy

        transfer_loss = self.forward_discrepancy(y_s, y_s_adv, labels_s, y_t, y_t_adv, epoch)

        return  0.25 * source_empirical_error, transfer_loss
                
        # self.beta1 * 0.25 * target_discrepancy - self.beta2 * 0.5 * source_discrepancy

    def forward_discrepancy(self, y_s: torch.Tensor, y_s_adv: torch.Tensor, labels_s: torch.Tensor, 
                y_t: torch.Tensor, y_t_adv: torch.Tensor, labels_t = None, epoch=0) -> torch.Tensor:
        source_discrepancy = self.source_discrepency(y_s, labels_s, y_s_adv)
        transfer_loss = -self.beta2 * 0.5 * source_discrepancy
        if epoch >= self.warm_up_epoch:
            target_discrepancy = self.target_discrepancy_loss(y_t, y_t_adv, epoch)  
            transfer_loss += self.beta1 * 0.25 * target_discrepancy

        return transfer_loss
