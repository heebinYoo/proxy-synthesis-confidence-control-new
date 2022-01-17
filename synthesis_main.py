import os
import sys
import glob
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

import net
import loss
import utils
from net.simpleFeatureExtractor import SimpleFeatureExtractor
from utils import sampler
from utils.synthesis_dataset import SynthesisDataset

from visualizer.visualizer import get_embed, generate_weight_embedding_relation_heatmap_figure, \
    generate_weight_tsne_figure, generate_feature_radius_dist_fig, generate_singular_value_figure, plot_synthesis_input, \
    plot_synthesis_embedding, plot_mnist_embedding

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data_name', default=None, type=str,
                    help='dataset name')
parser.add_argument('--save_path', default=None, type=str,
                    help='where your models will be saved')
parser.add_argument('--check_epoch', default=5, type=int,
                    help='do eval every check_epoch')
parser.add_argument('-j', '--workers', default=5, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    help='mini-batch size')
parser.add_argument('--modellr', default=0.0001, type=float,
                    help='initial model learning rate')
parser.add_argument('--centerlr', default=0.01, type=float,
                    help='initial center learning rate')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    help='weight decay', dest='weight_decay')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--eps', default=0.01, type=float,
                    help='epsilon for Adam')
parser.add_argument('--decay_rate', default=0.1, type=float,
                    help='decay rate')
parser.add_argument('--decay_step', default=20, type=int,
                    help='decay step')
parser.add_argument('--decay_stop', default=100000, type=int,
                    help='decay stop')
parser.add_argument('--dim', default=2, type=int,
                    help='dimensionality of embeddings')
parser.add_argument('--freeze_BN', action='store_true',
                    help='freeze bn')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='adam | adamw')
parser.add_argument('--clip_grad', default=0, type=int,
                    help='1: turn-on clip_grad, 0: turn-off clip_grad')
parser.add_argument('--input_size', default=224, type=int,
                    help='the size of input batch')
parser.add_argument('--do_nmi', action='store_true',
                    help='do nmi or not')
parser.add_argument('--n_instance', default=1, type=int,
                    help='n_instance')
parser.add_argument('--early_stop_epoch', default=0, type=int,
                    help='Early stop if there is no performance increase for such epochs')
parser.add_argument('--deterministic', default=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    help='Deterministic experiments')
parser.add_argument('--loss', default='SoftMax_vanilla', type=str,
                    help='loss you want')
parser.add_argument('--scale', default=1.0, type=float,
                    help='scale for softmax variations')
parser.add_argument('--ps_mu', default=0.0, type=float,
                    help='generation ratio in proxy synthesis')
parser.add_argument('--ps_alpha', default=0.0, type=float,
                    help='alpha for beta distribution in proxy synthesis')
parser.add_argument('--normalize', default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    help='do normalize?')
parser.add_argument('--confidence_control_mode', default="non", type=str,
                    help='what confidence_control_mode?')
parser.add_argument('--visualize', default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    help='visualize?')


def main():
    args = parser.parse_args()
    args.C = 2
    writer = SummaryWriter(args.save_path)
    device = torch.device("cuda", args.gpu) if torch.cuda.is_available() else torch.device("cpu")

    if args.deterministic:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        np.random.seed(0)
        torch.backends.cudnn.benchmark = False
        random.seed(0)
    train_dataset = SynthesisDataset(train=True, data_type=args.data_name)
    train_dataset.n_instance = args.n_instance
    train_sampler = sampler(train_dataset, args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler=train_sampler,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               pin_memory=True)
    test_dataset = SynthesisDataset(train=False, data_type=args.data_name)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    model = SimpleFeatureExtractor(feature_dim=args.dim, model_type="synthesis")
    model = model.to(device=device)
    criterion = loss.Norm_SoftMax(args.dim, args.C, scale=args.scale,
                                  ps_mu=args.ps_mu, ps_alpha=args.ps_alpha).cuda(args.gpu)

    params_list = [{"params": model.parameters(), "lr": args.modellr},
                   {"params": criterion.parameters(), "lr": args.centerlr}]

    if args.optimizer.lower() == 'Adam'.lower():
        optimizer = torch.optim.Adam(params_list, eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'AdamW'.lower():
        optimizer = torch.optim.AdamW(params_list, eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'RMSprop'.lower():
        optimizer = torch.optim.RMSprop(params_list, alpha=0.99, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer.lower() == 'SGD'.lower():
        optimizer = torch.optim.SGD(params_list, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    else:
        print("please specify optimizer")
        exit()

    if not args.deterministic:
        cudnn.benchmark = True

    global_step = 0
    for epoch in range(0, args.epochs):
        epoch += 1
        print('Training in Epoch[{}]'.format(epoch))
        adjust_learning_rate(optimizer, epoch, args)

        model.train()
        criterion.train()
        global_step = train(train_loader, model, criterion, optimizer, writer, global_step,
                            epoch, args)
        if epoch % args.check_epoch == 0 and args.visualize:
            model.eval()
            criterion.eval()

            train_embedding_list, train_label_list = get_embed(model, None, None, device, train_loader,
                                                               args.dim, args.batch_size,
                                                               len(train_loader.dataset), True)
            test_embedding_list, test_label_list = get_embed(model, None, None, device, test_loader, args.dim,
                                                             test_loader.batch_size,
                                                             len(test_loader.dataset), True)
            if not args.dim == 2:
                print("we can't plot synthesis data, make args.dim == 2")
                exit()
            input_fig = plot_synthesis_input(model, device, args.dim, train_loader, test_loader)
            writer.add_figure('synthesis-input-space', input_fig, epoch)
            embed_fig = plot_synthesis_embedding(model, device, args.dim, train_embedding_list, train_label_list,
                                                 test_embedding_list, test_label_list)
            writer.add_figure('embedding-space', embed_fig, epoch)


def train(train_loader, model, criterion, optimizer, writer, global_step, epoch, args):

    total_iter = len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        if args.ps_mu > 0.0:
            len_target = len(target)
            n_try = 50
            for _ in range(n_try):
                idx_list = list(range(len_target))
                swap_cnt = 0
                for now_ in range(len_target):
                    next_ = (now_ + 1) % len_target
                    now_idx = idx_list[now_]
                    next_idx = idx_list[next_]
                    now_t = target[now_idx].item()
                    next_t = target[next_idx].item()
                    if now_t == next_t:
                        next_next_ = (next_ + 1) % len_target
                        idx_list = swap_idx(idx_list, next_, next_next_)
                        swap_cnt += 1
                input = input[idx_list]
                target = target[idx_list]
                if swap_cnt == 0:
                    break
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        output = model(input)
        loss = criterion(output, target)

        if i % 10 == 0:
            print('[%d/%d] loss: %.4f' % (i + 1, total_iter, loss.item()))
            writer.add_scalar('train/loss', loss, global_step)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.flush()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad == 1:
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)
        optimizer.step()

        global_step += 1

    return global_step


def swap_idx(array, now_, next_):
    tmp = array[now_]
    array[now_] = array[next_]
    array[next_] = tmp

    return array


def adjust_learning_rate(optimizer, epoch, args):
    if epoch % args.decay_step == 0 and epoch <= args.decay_stop:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.decay_rate
            print(param_group['lr'])


if __name__ == '__main__':
    main()
