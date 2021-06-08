# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2020-12-08 19:46:19
"""

import argparse
import random
import importlib
import platform

import numpy as np

import torch
from torch import nn
from torchvision import models

from Train import pretrain
from utils import global_variable as GV
import os

def display_args(args):
    print('===== task arguments =====')
    print('data_name = %s' % (args.data_name))
    print('network_name = %s' % (args.network_name))
    print('===== experiment environment arguments =====')
    print('devices = %s' % (str(args.devices)))
    print('flag_debug = %r' % (args.flag_debug))
    print('n_workers = %d' % (args.n_workers))
    print('===== optimizer arguments =====')
    print('lr = %f' % (args.lr))
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
    print('n_training_epochs = %d' % (args.n_training_epochs))
    print('batch_size = %d' % (args.batch_size))



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
    parser.add_argument('--n_new_classes', type=int, default=0)
    parser.add_argument('--network_name', type=str, default='wide_resnet', choices=['resnet', 'wide_resnet', 'mobile_net'])
    # experiment environment arguments
    parser.add_argument('--devices', type=int, nargs='+', default=GV.DEVICES)
    parser.add_argument('--flag_debug', action='store_true', default=False)
    parser.add_argument('--n_workers', type=int, default=GV.WORKERS)
    # optimizer arguments
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--point', type=int, nargs='+', default=(50,100,150))
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--wd', type=float, default=0.0005)  # weight decay
    parser.add_argument('--mo', type=float, default=0.9)  # momentum
    # network arguments
    parser.add_argument('--depth', type=int, default=40)
    parser.add_argument('--width', type=int, default=2)
    parser.add_argument('--ca', type=float, default=1)  # channel
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    # training procedure arguments
    parser.add_argument('--n_training_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()

    display_args(args)

    data_path = 'datasets/' + args.data_name + '/'

    Data = importlib.import_module('dataloaders.' + args.data_name)
    Network = importlib.import_module('networks.' + args.network_name)

    # generate data_loader
    train_data_loader = Data.generate_data_loader(data_path, 'train', args.n_new_classes, args.batch_size, args.n_workers)
    args.number_of_classes = train_data_loader.dataset.get_n_classes()
    print('===== train data loader ready. =====')
    test_data_loader = Data.generate_data_loader(data_path, 'test', args.n_new_classes, args.batch_size, args.n_workers)
    print('===== test data loader ready. =====')

    # generate network
    network = Network.MyNetwork(args)
    network = network.cuda(args.devices[0])
    if len(args.devices) > 1:
        network = torch.nn.DataParallel(network, device_ids=args.devices)
    print('===== network ready. =====')

    model_save_path = 'saves/pretrained_teachers/' + \
                        args.data_name + '_' + args.network_name + \
                        '_lr=' + str(args.lr) + \
                        '_point=' + str(args.point) + \
                        '_gamma=' + str(args.gamma) + \
                        '_wd=' + str(args.wd) + \
                        '_mo=' + str(args.mo) + \
                        '.model'
    statistics_save_path = 'saves/teacher_statistics/' + \
                                        args.data_name + '_' + args.network_name + \
                                        '_lr=' + str(args.lr) + \
                                        '_point=' + str(args.point) + \
                                        '_gamma=' + str(args.gamma) + \
                                        '_wd=' + str(args.wd) + \
                                        '_mo=' + str(args.mo) + \
                                        '.stat'

    # create model directories
    dirs = os.path.dirname(model_save_path)
    os.makedirs(dirs, exist_ok=True)

    # model training
    training_loss_list, training_accuracy_list, testing_accuracy_list = \
        pretrain(args, train_data_loader, test_data_loader, network, model_save_path)
    record = {
        'training_loss': training_loss_list,
        'training_accuracy': training_accuracy_list,
        'testing_accuracy': testing_accuracy_list
    }

    # create stats directories
    dirs = os.path.dirname(statistics_save_path)
    os.makedirs(dirs, exist_ok=True)
    if args.n_training_epochs > 0 and (not args.flag_debug):
        torch.save(record, statistics_save_path)
    print('===== pretraining finish. =====')

    # load best model
    if not args.flag_debug:
        record = torch.load(model_save_path)
        best_testing_accuracy = record['testing_accuracy']
        network.load_state_dict(record['state_dict'])
        print('===== best model loaded, testing acc = %f. =====' % (record['testing_accuracy']))