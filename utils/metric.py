# -*- coding: utf-8 -*-
"""
@Author Su Lu

@Date: 2021-07-20 14:19:29
"""

import argparse
import random
import importlib
import platform
import copy
import sys
sys.path.append('..')

import numpy as np

import torch
from torch import nn
from torchvision import models

from utils import global_variable as GV
import os

def display_args(args):
    print('===== task arguments =====')
    print('data_name = %s' % (args.data_name))
    print('n_classes = %d' % (args.n_classes))
    print('n_new_classes = %d' % (args.n_new_classes))
    print('teacher_network_name = %s' % (args.teacher_network_name))
    print('student_network_name = %s' % (args.student_network_name))
    print('model_name = %s' % (args.model_name))
    print('===== experiment environment arguments =====')
    print('devices = %s' % (str(args.devices)))
    print('flag_debug = %r' % (args.flag_debug))
    print('n_workers = %d' % (args.n_workers))
    print('===== optimizer arguments =====')
    print('lr1 = %f' % (args.lr1))
    print('lr2 = %f' % (args.lr2))
    print('point = %s' % str((args.point)))
    print('gamma = %f' % (args.gamma))
    print('weight_decay = %f' % (args.wd))
    print('momentum = %f' % (args.mo))
    print('===== network arguments =====')
    print('depth = %d' % (args.depth))
    print('width = %d' % (args.width))
    print('ca = %f' % (args.ca))
    print('dropout_rate = %d' % (args.dropout_rate))
    print('===== training procedure arguments =====')
    print('n_training_epochs1 = %d' % (args.n_training_epochs1))
    print('n_training_epochs2 = %d' % (args.n_training_epochs2))
    print('batch_size = %d' % (args.batch_size))
    print('tau1 = %f' % (args.tau1))
    print('tau2 = %f' % (args.tau2))
    print('lambd = %f' % (args.lambd))



def compute_metric(args, teacher, teacher_data_loader, train_data_loader, metric_save_path):
    teacher_labels = []
    for _, batch in enumerate(teacher_data_loader):
        _, _, labels = batch
        labels = labels.long().numpy()
        teacher_labels += list(labels)
    teacher_labels = np.unique(np.array(teacher_labels))

    inout = []
    weight = []
    for _, batch in enumerate(train_data_loader):
        images, _, labels = batch
        images = images.float().cuda(args.devices[0])
        labels = labels.long().numpy()

        for label in labels:
            if label in teacher_labels:
                inout.append(1)
            else:
                inout.append(0)

        with torch.no_grad():
            teacher_output_logits, teacher_embeddings = teacher.forward(images, flag_both=True)
            teacher_pseudo_labels = torch.argmax(teacher_output_logits, dim=1)
            weights = nn.CrossEntropyLoss(reduction='none')(teacher_output_logits, teacher_pseudo_labels)
            weights = (weights - torch.min(weights)) / (torch.max(weights) - torch.min(weights))
            weights = 2 * torch.sigmoid(-1 * weights)
            weights = weights.cpu().numpy()
            weight += list(weights)
        
    inout = np.array(inout)
    weight = np.array(weight)

    inout_save_path = metric_save_path + \
        args.data_name + '_inout' + \
        '_newclass=' + str(args.n_new_classes) + '.npy'
    weight_save_path = metric_save_path + \
        args.data_name + '_weight' + \
        '_newclass=' + str(args.n_new_classes) + '.npy'
    
    np.save(inout_save_path, inout)
    np.save(weight_save_path, weight)



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
    parser.add_argument('--n_new_classes', type=int, default=20)
    parser.add_argument('--teacher_network_name', type=str, default='wide_resnet', choices=['resnet', 'wide_resnet', 'mobile_net'])
    parser.add_argument('--student_network_name', type=str, default='wide_resnet', choices=['resnet', 'wide_resnet', 'mobile_net'])
    parser.add_argument('--model_name', type=str, default='wgkd', choices=['ce', 'kd', 'gkd', 'wgkd'])
    # experiment environment arguments
    parser.add_argument('--devices', type=int, nargs='+', default=GV.DEVICES)
    parser.add_argument('--flag_debug', action='store_true', default=False)
    parser.add_argument('--n_workers', type=int, default=GV.WORKERS)
    # optimizer arguments
    parser.add_argument('--lr1', type=float, default=0.1)
    parser.add_argument('--lr2', type=float, default=0.1)
    parser.add_argument('--point', type=int, nargs='+', default=(50,100,150))
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--wd', type=float, default=0.0005)  # weight decay
    parser.add_argument('--mo', type=float, default=0.9)  # momentum
    # network arguments
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--ca', type=float, default=0.25)  # channel
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    # training procedure arguments
    parser.add_argument('--n_training_epochs1', type=int, default=100)
    parser.add_argument('--n_training_epochs2', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--tau1', type=float, default=4) # temperature in stage 1
    parser.add_argument('--tau2', type=float, default=1) # temperature in stage 2
    parser.add_argument('--lambd', type=float, default=1) # weight of teaching loss in stage 2

    args = parser.parse_args()

    display_args(args)

    data_path = '../datasets/' + args.data_name + '/'
    if args.data_name == 'CIFAR-100':
        assert(args.n_classes <= 100)
        assert(args.n_new_classes <= 50)
    elif args.data_name == 'CUB-200':
        assert(args.n_classes <= 200)
        assert(args.n_new_classes <= 100)

    # import modules
    Data = importlib.import_module('dataloaders.' + args.data_name)
    Network_Teacher = importlib.import_module('networks.' + args.teacher_network_name)
    Network_Student = importlib.import_module('networks.' + args.student_network_name)

     # generate data_loader
    train_data_loader = Data.generate_data_loader(data_path, 'train', args.n_classes, args.n_new_classes, args.batch_size, args.n_workers)
    print('===== train data loader ready. =====')
    teacher_data_loader = Data.generate_data_loader(data_path, 'train', args.n_classes, 0, args.batch_size, args.n_workers)
    print('==== teacher data loader ready. =====')


    # generate teacher network
    if args.model_name != 'ce':
        if args.teacher_network_name == 'resnet':
            teacher_args = copy.copy(args)
            teacher_args.depth = 110
            teacher = Network_Teacher.MyNetwork(teacher_args)
            pretrained_teacher_save_path = '../saves/pretrained_teachers/' + args.data_name + '_resnet' + \
                '_class=' + str(args.n_classes) + '_teacher.model'
        elif args.teacher_network_name == 'wide_resnet':
            teacher_args = copy.copy(args)
            teacher_args.depth, teacher_args.width = 40, 2
            teacher = Network_Teacher.MyNetwork(teacher_args)
            pretrained_teacher_save_path = '../saves/pretrained_teachers/' + args.data_name + '_wide_resnet' + \
                '_class=' + str(args.n_classes) + '_teacher.model'
        elif args.teacher_network_name == 'mobile_net':
            teacher_args = copy.copy(args)
            teacher_args.ca = 1.0
            teacher = Network_Teacher.MyNetwork(teacher_args)
            pretrained_teacher_save_path = '../saves/pretrained_teachers/' + args.data_name + '_mobile_net' + \
                '_class=' + str(args.n_classes) + '_teacher.model'
        record = torch.load(pretrained_teacher_save_path, map_location='cpu')
        teacher.load_state_dict(record['state_dict'])
        teacher = teacher.cuda(args.devices[0])
        if len(args.devices) > 1:
            teacher = torch.nn.DataParallel(teacher, device_ids=args.devices)
        # set teacher to evaluation mode
        teacher.eval()
    else:
        teacher = None
    print('===== teacher ready. =====')

    metric_save_path = '../saves/metrics/'

    compute_metric(args, teacher, teacher_data_loader, train_data_loader, metric_save_path)