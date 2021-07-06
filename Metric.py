# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2021-06-07 19:15:01
"""

import argparse
import random
import importlib
import platform
import copy

import numpy as np

import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torchvision import models

from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

from networks import resnet, wide_resnet, mobile_net

from Train import train_stage1
from Train import train_stage2
from utils import global_variable as GV
import os

def display_args(args):
    print('===== task arguments =====')
    print('data_name = %s' % (args.data_name))
    print('n_classes = %d' % (args.n_classes))
    print('n_new_classes = %d' % (args.n_new_classes))
    print('teacher_network_name = %s' % (args.teacher_network_name))
    print('===== experiment environment arguments =====')
    print('devices = %s' % (str(args.devices)))
    print('flag_debug = %r' % (args.flag_debug))
    print('n_workers = %d' % (args.n_workers))



def get_instance_metric(args, train_data_loader, teacher, teacher_label_set):
    maxlogit_all = []
    entropy_all = []
    pseudo_all = []
    inout_all = []

    for _, batch in enumerate(train_data_loader):
        images, labels, raw_labels = batch
        images = images.float().cuda(args.devices[0])
        labels = labels.long().cuda(args.devices[0])
        raw_labels = np.array(raw_labels)

        with torch.no_grad():
            teacher_logits = teacher.forward(images)
            maxlogit = torch.max(teacher_logits, dim=1)[0]
            distributions = Categorical(logits=teacher_logits)
            entropy = distributions.entropy()
            pseudo_labels = torch.argmax(teacher_logits, dim=1)
            pseudo = nn.CrossEntropyLoss(reduction='none')(teacher_logits, pseudo_labels)

            maxlogit_all += list(maxlogit.cpu().numpy())
            entropy_all += list(entropy.cpu().numpy())
            for raw_label in raw_labels:
                if raw_label in teacher_label_set:
                    inout_all.append(1)
                else:
                    inout_all.append(0)
            pseudo_all += list(pseudo.cpu().numpy())

    maxlogit_all = np.array(maxlogit_all)
    entropy_all = np.array(entropy_all)
    pseudo_all = np.array(pseudo_all)
    inout_all = np.array(inout_all)

    maxlogit_path = 'saves/metrics/maxlogit_' + args.data_name + '_' + str(args.n_new_classes) + '.npy'
    np.save(maxlogit_path, maxlogit_all)
    entropy_path = 'saves/metrics/entropy_' + args.data_name + '_' + str(args.n_new_classes) + '.npy'
    np.save(entropy_path, entropy_all)
    pseudo_path = 'saves/metrics/pseudo_' + args.data_name + '_' + str(args.n_new_classes) + '.npy'
    np.save(pseudo_path, pseudo_all)
    inout_path = 'saves/metrics/inout_' + args.data_name + '_' + str(args.n_new_classes) + '.npy'
    np.save(inout_path, inout_all)



def draw_instance_metric(args):
    maxlogit_path = 'saves/metrics/maxlogit_' + args.data_name + '_' + str(args.n_new_classes) + '.npy'
    entropy_path = 'saves/metrics/entropy_' + args.data_name + '_' + str(args.n_new_classes) + '.npy'
    pseudo_path = 'saves/metrics/pseudo_' + args.data_name + '_' + str(args.n_new_classes) + '.npy'
    inout_path = 'saves/metrics/inout_' + args.data_name + '_' + str(args.n_new_classes) + '.npy'

    maxlogit = np.load(maxlogit_path)
    maxlogit = (maxlogit - np.min(maxlogit)) / (np.max(maxlogit) - np.min(maxlogit))
    entropy = np.load(entropy_path)
    entropy = (entropy - np.min(entropy)) / (np.max(entropy) - np.min(entropy))
    pseudo = np.load(pseudo_path)
    pseudo = (pseudo - np.min(pseudo)) / (np.max(pseudo) - np.min(pseudo))
    inout = np.load(inout_path)

    in_indexes = np.where(inout == 1)[0]
    out_indexes = np.where(inout == 0)[0]

    # maxlogit
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(1, 1, 1)

    ax.set_xlabel('maximum logit', fontsize=28)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize': 28})

    ax.set_ylabel('number of instances ($\\times 10^3$)', fontsize=28)
    ax.set_ylim(0, 3000)
    ax.set_yticks([0, 1000, 2000, 3000])
    ax.set_yticklabels([0, 1, 2, 3], fontdict={'fontsize': 28})

    ax.hist(maxlogit[in_indexes], bins=30, label='seen classes',
        color='red', edgecolor='black', alpha=0.5, linewidth=2)
    ax.hist(maxlogit[out_indexes], bins=30, label='unseen classes',
        color='blue', edgecolor='black', alpha=0.5, linewidth=2)

    ax.legend(prop={'size': 28})
    plt.savefig('saves/metrics/figures/maxlogit_' + args.data_name + '_' + str(args.n_new_classes) + '.jpg')
    plt.savefig('saves/metrics/figures/maxlogit_' + args.data_name + '_' + str(args.n_new_classes) + '.pdf')
    plt.close()

    # entropy
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(1, 1, 1)

    ax.set_xlabel('entropy', fontsize=28)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize': 28})

    ax.set_ylabel('number of instances ($\\times 10^3$)', fontsize=28)
    ax.set_ylim(0, 10000)
    ax.set_yticks([0, 2000, 4000, 6000, 8000, 10000])
    ax.set_yticklabels([0, 2, 4, 6, 8, 10], fontdict={'fontsize': 28})

    ax.hist(entropy[in_indexes], bins=30, label='seen classes',
        color='red', edgecolor='black', alpha=0.5, linewidth=2)
    ax.hist(entropy[out_indexes], bins=30, label='unseen classes',
        color='blue', edgecolor='black', alpha=0.5, linewidth=2)
        
    ax.legend(prop={'size': 28})
    plt.savefig('saves/metrics/figures/entropy_' + args.data_name + '_' + str(args.n_new_classes) + '.jpg')
    plt.savefig('saves/metrics/figures/entropy_' + args.data_name + '_' + str(args.n_new_classes) + '.pdf')
    plt.close()

    # pseudo
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(1, 1, 1)

    ax.set_xlabel('pseudo', fontsize=28)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize': 28})

    ax.set_ylabel('number of instances ($\\times 10^3$)', fontsize=28)
    ax.set_ylim(0, 10000)
    ax.set_yticks([0, 2000, 4000, 6000, 8000, 10000])
    ax.set_yticklabels([0, 2, 4, 6, 8, 10], fontdict={'fontsize': 28})

    ax.hist(pseudo[in_indexes], bins=30, label='seen classes',
        color='red', edgecolor='black', alpha=0.5, linewidth=2)
    ax.hist(pseudo[out_indexes], bins=30, label='unseen classes',
        color='blue', edgecolor='black', alpha=0.5, linewidth=2)
        
    ax.legend(prop={'size': 28})
    plt.savefig('saves/metrics/figures/pseudo_' + args.data_name + '_' + str(args.n_new_classes) + '.jpg')
    plt.savefig('saves/metrics/figures/pseudo_' + args.data_name + '_' + str(args.n_new_classes) + '.pdf')
    plt.close()



def test_instance_metric(args):
    maxlogit_path = 'maxlogit_' + args.data_name + '_' + str(args.n_new_classes) + '.npy'
    entropy_path = 'entropy_' + args.data_name + '_' + str(args.n_new_classes) + '.npy'
    pseudo_path = 'pseudo_' + args.data_name + '_' + str(args.n_new_classes) + '.npy'
    inout_path = 'inout_' + args.data_name + '_' + str(args.n_new_classes) + '.npy'

    maxlogit = np.load(maxlogit_path)
    maxlogit = (maxlogit - np.min(maxlogit)) / (np.max(maxlogit) - np.min(maxlogit))
    entropy = np.load(entropy_path)
    entropy = (entropy - np.min(entropy)) / (np.max(entropy) - np.min(entropy))
    pseudo = np.load(pseudo_path)
    pseudo = (pseudo - np.min(pseudo)) / (np.max(pseudo) - np.min(pseudo))
    inout = np.load(inout_path)

    print(roc_auc_score(inout, maxlogit))
    print(roc_auc_score(1 - inout, entropy))
    print(roc_auc_score(1 - inout, pseudo))


if __name__ == '__main__':
    # set random seed
    random.seed(960402)
    np.random.seed(960402)
    torch.manual_seed(960402)
    torch.cuda.manual_seed(960402)
    torch.backends.cudnn.deterministic = True

    # create a parser
    parser = argparse.ArgumentParser()
    # task arguments
    parser.add_argument('--data_name', type=str, default='CIFAR-100', choices=['CIFAR-100', 'CUB-200'])
    parser.add_argument('--n_classes', type=int, default=50)
    parser.add_argument('--n_new_classes', type=int, default=10)
    parser.add_argument('--teacher_network_name', type=str, default='wide_resnet', choices=['resnet', 'wide_resnet', 'mobile_net'])
    # experiment environment arguments
    parser.add_argument('--devices', type=int, nargs='+', default=GV.DEVICES)
    parser.add_argument('--flag_debug', action='store_true', default=False)
    parser.add_argument('--n_workers', type=int, default=GV.WORKERS)
     # network arguments
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--ca', type=float, default=0.25)  # channel
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    # training procedure arguments
    parser.add_argument('--batch_size', type=int, default=256)

    args = parser.parse_args()

    display_args(args)

    data_path = 'datasets/' + args.data_name + '/'
    if args.data_name == 'CIFAR-100':
        assert(args.n_classes <= 100)
        assert(args.n_new_classes <= 50)
    elif args.data_name == 'CUB-200':
        assert(args.n_classes <= 200)
        assert(args.n_new_classes <= 100)

    # import modules
    Data = importlib.import_module('dataloaders.' + args.data_name)
    Network_Teacher = importlib.import_module('networks.' + args.teacher_network_name)

    # generate data_loader
    train_data_loader = Data.generate_data_loader(data_path, 'train', args.n_classes, args.n_new_classes, args.batch_size, args.n_workers)
    print('===== train data loader ready. =====')
    teacher_data_loader = Data.generate_data_loader(data_path, 'train', args.n_classes, 0, args.batch_size, args.n_workers)
    print('===== teacher data loader ready. =====')
    teacher_label_set = np.unique(teacher_data_loader.dataset.labels)

    if args.teacher_network_name == 'resnet':
        teacher_args = copy.copy(args)
        teacher_args.depth = 110
        teacher = Network_Teacher.MyNetwork(teacher_args)
        pretrained_teacher_save_path = 'saves/pretrained_teachers/' + args.data_name + '_resnet' + \
            '_class=' + str(args.n_classes) + '_teacher.model'
    elif args.teacher_network_name == 'wide_resnet':
        teacher_args = copy.copy(args)
        teacher_args.depth, teacher_args.width = 40, 2
        teacher = Network_Teacher.MyNetwork(teacher_args)
        pretrained_teacher_save_path = 'saves/pretrained_teachers/' + args.data_name + '_wide_resnet' + \
            '_class=' + str(args.n_classes) + '_teacher.model'
    elif args.teacher_network_name == 'mobile_net':
        teacher_args = copy.copy(args)
        teacher_args.ca = 1.0
        teacher = Network_Teacher.MyNetwork(teacher_args)
        pretrained_teacher_save_path = 'saves/pretrained_teachers/' + args.data_name + '_mobile_net' + \
            '_class=' + str(args.n_classes) + '_teacher.model'
    record = torch.load(pretrained_teacher_save_path, map_location='cpu')
    teacher.load_state_dict(record['state_dict'])
    teacher = teacher.cuda(args.devices[0])
    if len(args.devices) > 1:
        teacher = torch.nn.DataParallel(teacher, device_ids=args.devices)
    # set teacher to evaluation mode
    teacher.eval()
    print('===== teacher ready. =====')

    # get_instance_metric(args, train_data_loader, teacher, teacher_label_set)
    draw_instance_metric(args)
    # test_instance_metric(args)