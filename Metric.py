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
from torchvision import models

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