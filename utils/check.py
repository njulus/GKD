# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2021-06-10 13:03:33
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



def check_model(args):
    model_save_path2 = '../saves/trained_students/' + \
                        args.data_name + '_' + args.model_name + '_' + args.student_network_name + '_' + args.teacher_network_name + \
                        '_class=' + str(args.n_classes) + \
                        '_newclass=' + str(args.n_new_classes) + \
                        '_lr2=' + str(args.lr2) + \
                        '_point=' + str(args.point) + \
                        '_gamma=' + str(args.gamma) + \
                        '_wd=' + str(args.wd) + \
                        '_mo=' + str(args.mo) + \
                        '_depth=' + str(args.depth) + \
                        '_width=' + str(args.width) + \
                        '_ca=' + str(args.ca) + \
                        '_tau2=' + str(args.tau2) + \
                        '_lambd=' + str(args.lambd) + \
                        '.model'
    record = torch.load(model_save_path2, map_location='cpu')
    print('===== best model in stage 2 loaded, testing acc = %f. =====' % (record['testing_accuracy']))



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
    parser.add_argument('--data_name', type=str, default='CUB-200', choices=['CIFAR-100', 'CUB-200'])
    parser.add_argument('--n_classes', type=int, default=100)
    parser.add_argument('--n_new_classes', type=int, default=0)
    parser.add_argument('--teacher_network_name', type=str, default='wide_resnet', choices=['resnet', 'wide_resnet', 'mobile_net'])
    parser.add_argument('--student_network_name', type=str, default='mobile_net', choices=['resnet', 'wide_resnet', 'mobile_net'])
    parser.add_argument('--model_name', type=str, default='ce', choices=['ce', 'kd', 'gkd'])
    # experiment environment arguments
    parser.add_argument('--devices', type=int, nargs='+', default=GV.DEVICES)
    parser.add_argument('--flag_debug', action='store_true', default=False)
    parser.add_argument('--n_workers', type=int, default=GV.WORKERS)
    # optimizer arguments
    parser.add_argument('--lr1', type=float, default=0.1)
    parser.add_argument('--lr2', type=float, default=0.1)
    parser.add_argument('--point', type=int, nargs='+', default=(50,100,150))
    parser.add_argument('--gamma', type=float, default=-1.0)
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
    parser.add_argument('--tau2', type=float, default=2) # temperature in stage 2
    parser.add_argument('--lambd', type=float, default=100) # weight of teaching loss in stage 2

    args = parser.parse_args()

    display_args(args)

    check_model(args)