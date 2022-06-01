import random
import time
import warnings
import argparse
import os.path as osp
import shutil
import numpy as np 
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F

from models.resnet import resnet50
from dalib import AUCDomainAdapation
from dataloaders import load_dataloader, get_dataset_names, ForeverDataIterator
from modules import CompleteLogger, ImageClassifier, AverageMeter, ProgressMeter
from metrics import auc_mp_fast
from modules.grl import GradientReverseLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_support_DA = {
    'AUC_DA': AUCDomainAdapation
}
def main(args):

    global device
    device = torch.device("cuda:{gpu}".format(gpu=args.gpu) if torch.cuda.is_available() else "cpu")

    if not isinstance(args.seed, list):
        args.seed = [args.seed]

    base_path = osp.join(args.root, 
                        args.data_save, 
                        args.data, 
                        args.da_method, 
                        '->'.join([args.source[0].upper(), args.target[0].upper()]),
                        'lr_{}'.format(args.lr),
                        'epsilon_{}'.format(args.epsilon),
                        'loss_type_{}'.format(args.loss_type),
                        'warm_up_epoch_{}'.format(args.warm_up_epoch))

    model_name = osp.join('_'.join(['lr_{}'.format(args.lr),
                        'epsilon_{}'.format(args.epsilon),
                        'loss_type_{}'.format(args.loss_type),
                        'warm_up_epoch_{}'.format(args.warm_up_epoch)]))
    if not osp.exists(base_path):
        import os 
        os.makedirs(base_path)
    logger = CompleteLogger(osp.join(base_path, 'saved_model_all_seeds'))

    total_mauc = []
    for random_seed in args.seed:
        logger.info("======> Training with random seed: {}".format(random_seed))
        if args.seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')

        cudnn.benchmark = False

        # get dataset and dataloader
        train_source_dataset, train_target_dataset, val_loader, num_classes, probs = \
            load_dataloader(args)

        print(probs)

        train_source_iter = ForeverDataIterator(train_source_dataset)
        train_target_iter = ForeverDataIterator(train_target_dataset)


        pool_layer = nn.Identity() if args.no_pool else None

        backbone = resnet50(pretrained=False).cuda()
        classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                    width=args.bottleneck_dim, pool_layer=pool_layer, grl=GradientReverseLayer()).cuda()

        try:
            da_method = all_support_DA[args.da_method](num_classes, probs, **vars(args)).cuda()
        except KeyError as e:
            raise e('Do not support da_method {}'.format(args.da_method))
        
        optimizer = SGD(classifier.get_parameters(base_lr = args.lr), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=False)


        lr_scheduler = ExponentialLR(optimizer, gamma = args.lr_decay)
        best_mauc = 0.
        best_epoch = 0

        logger.info(args)

        for epoch in range(args.epochs):
            train(args, train_source_iter, train_target_iter, classifier, da_method, optimizer, lr_scheduler, epoch)

            # evaluate on train set
            train_mauc = validate(train_source_dataset, classifier)
            logger.info("cur_train_mauc = {:.4f}".format(train_mauc))
            
            # evaluate on validation set
            mauc = validate(val_loader, classifier)

            cur_checkpoint = {
                'classifier': classifier.state_dict(),
                'schedular': lr_scheduler.state_dict(),
                'epoch': epoch
            }

            torch.save(cur_checkpoint, logger.get_checkpoint_path('{}_latest_seed_{}'.format(model_name, random_seed)))

            if mauc > best_mauc:
                shutil.copy(logger.get_checkpoint_path('{}_latest_seed_{}'.format(model_name, random_seed)), 
                            logger.get_checkpoint_path('{}_best_seed_{}'.format(model_name, random_seed)))
                best_epoch = epoch
            
            best_mauc = max(best_mauc, mauc)

            logger.info("Epoch: {}/{}, MAUC: {}, cur_best_MAUC: {} at Epoch: {}".format(epoch, 
                                                                                        args.epochs, 
                                                                                        mauc,
                                                                                        best_mauc,
                                                                                        best_epoch))

        logger.info("======> Random seed: {}".format(random_seed))
        logger.info("best_val_mauc = {:.4f}".format(best_mauc))
        logger.info("======>")

        total_mauc.append(best_mauc)
    logger.info("======> Average performance over {} experiments: {}".format(len(args.seed), np.mean(total_mauc)))
    logger.info("======> Std performance over {} experiments: {}".format(len(args.seed), np.std(total_mauc)))


def train(args, train_source_iter, train_target_iter, classifier, da_method, optimizer,
          lr_scheduler, epoch):

    batch_time = AverageMeter('Time', ':.4f')
    data_time = AverageMeter('Data', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    trans_losses = AverageMeter('Trans Loss', ':.10f')
    
    progress = ProgressMeter(
            args.epochs,
            [batch_time, data_time, losses, trans_losses],
            prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    classifier.train()
    da_method.train()

    end = time.time()
    for i in range(args.iters_per_epoch):

        x_s, labels_s = next(train_source_iter)
        x_t, labels_t = next(train_target_iter)

        x_s = x_s.cuda()
        x_t = x_t.cuda()
        labels_s = labels_s.cuda()
        labels_t = labels_t.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        outputs, outputs_adv = classifier(x)
        y_s, y_t = outputs.chunk(2, dim=0)
        y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)

        # for adversarial classifier, minimize negative mdd is equal to maximize mdd
        cls_loss, transfer_loss = da_method(y_s, y_s_adv, labels_s, y_t, y_t_adv, epoch)
        loss = cls_loss + transfer_loss * args.trade_off

        losses.update(loss.item(), x_s.size(0))
        
        trans_losses.update(transfer_loss.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    lr_scheduler.step()

def validate(val_loader, model):

    # switch to evaluate mode
    model.eval()

    preds, ty = [], []

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)

            preds.append(F.softmax(output, dim = -1).cpu().numpy())
            ty.append(target.cpu().long().numpy())

        preds = np.concatenate(preds, axis = 0)
        ty = np.concatenate(ty, axis = 0)
    
    mauc = auc_mp_fast(ty, preds)

    return mauc

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MDD for Unsupervised Domain Adaptation')

    parser.add_argument('--root', metavar='DIR', default='data',
                        help='root path of dataset')
    
    parser.add_argument('--data_save', default='save', help='where to save model')

    parser.add_argument('-d', '--data', metavar='DATA', default='NSFW', choices=get_dataset_names(),
                        help='dataset: ' + ' | '.join(get_dataset_names()) +
                             ' (default: Office31)')

    parser.add_argument('-dm', '--da_method', help='method of DA to be run', type=str, default='AUC_DA')
    parser.add_argument('-s', '--source', help='source domain(s)', default='neutral_0.01', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', default='drawings_0.01', nargs='+')
    parser.add_argument('--gpu', default=0, type=int, help='gpu number')

    parser.add_argument('--bottleneck_dim', default=1024, type=int)
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--epsilon', type=float, default=0.0, help="AUC epsilon")
    parser.add_argument('--beta1', type=float, default=0.001, help="target discrepency beta")
    parser.add_argument('--beta2', type=float, default=0.001, help="source discrepency beta")
    parser.add_argument('--gamma', type=float, default=5.0, help="weighted function gamma")
    parser.add_argument('--loss_type', type=str, default='log_loss', help="AUC loss type")
    parser.add_argument('--trade-off', default=1.0, type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')

    parser.add_argument('--base_lr', default=1.0, type=float,
                        metavar='LR', help='first-step learning rate')

    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0002, type=float)
    parser.add_argument('--lr-decay', default=0.98, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=[0], type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    
    parser.add_argument('--use_pseudo_label', action='store_false',
                        help='whether adopt the pseudo label on the target domain')

    parser.add_argument('--decay_epoch', default=1, type=int,
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument('--warm_up_epoch', default=5, type=int,
                        help='whether output per-class accuracy during evaluation')
    args = parser.parse_args()

    main(args)
