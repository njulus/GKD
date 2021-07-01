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
    parser.add_argument('--teacher_network_name', type=str, default='mobile_net', choices=['resnet', 'wide_resnet', 'mobile_net'])
    parser.add_argument('--student_network_name', type=str, default='mobile_net', choices=['resnet', 'wide_resnet', 'mobile_net'])
    parser.add_argument('--model_name', type=str, default='gkd', choices=['ce', 'kd', 'gkd', 'pgkd'])
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
    parser.add_argument('--n_training_epochs1', type=int, default=0)
    parser.add_argument('--n_training_epochs2', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--tau1', type=float, default=4) # temperature in stage 1
    parser.add_argument('--tau2', type=float, default=1) # temperature in stage 2
    parser.add_argument('--lambd', type=float, default=10) # weight of teaching loss in stage 2

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
    Network_Student = importlib.import_module('networks.' + args.student_network_name)

    # generate data_loader
    train_data_loader = Data.generate_data_loader(data_path, 'train', args.n_classes, args.n_new_classes, args.batch_size, args.n_workers)
    print('===== train data loader ready. =====')
    test_data_loader = Data.generate_data_loader(data_path, 'test', args.n_classes, args.n_new_classes, args.batch_size, args.n_workers)
    print('===== test data loader ready. =====')

    # generate teacher network
    if args.model_name != 'ce':
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
    else:
        teacher = None
    print('===== teacher ready. =====')

    # generate student network
    student = Network_Student.MyNetwork(args)
    student = student.cuda(args.devices[0])
    if len(args.devices) > 1:
        student = torch.nn.DataParallel(student, device_ids=args.devices)
    print('===== student ready. =====')

    if args.model_name == 'gkd' or args.model_name == 'pgkd':
        # model save path and statistics save path for stage 1
        model_save_path1 = 'saves/trained_students/' + \
                            args.data_name + '_' + 'gkd' + '_' + args.student_network_name + '_' + args.teacher_network_name + \
                            '_class=' + str(args.n_classes) + \
                            '_newclass=' + str(args.n_new_classes) + \
                            '_lr1=' + str(args.lr1) + \
                            '_wd=' + str(args.wd) + \
                            '_mo=' + str(args.mo) + \
                            '_depth=' + str(args.depth) + \
                            '_width=' + str(args.width) + \
                            '_ca=' + str(args.ca) + \
                            '_tau1=' + str(args.tau1) + \
                            '.model'
        statistics_save_path1 = 'saves/student_statistics/' + \
                                args.data_name + '_' + 'gkd' + '_' + args.student_network_name + '_' + args.teacher_network_name + \
                                '_class=' + str(args.n_classes) + \
                                '_newclass=' + str(args.n_new_classes) + \
                                '_lr1=' + str(args.lr1) + \
                                '_wd=' + str(args.wd) + \
                                '_mo=' + str(args.mo) + \
                                '_depth=' + str(args.depth) + \
                                '_width=' + str(args.width) + \
                                '_ca=' + str(args.ca) + \
                                '_tau1=' + str(args.tau1) + \
                                '.stat'

        # create model directories
        dirs = os.path.dirname(model_save_path1)
        os.makedirs(dirs, exist_ok=True)

        # model training stage 1
        training_loss_list1, testing_accuracy_list1 = \
            train_stage1(args, train_data_loader, test_data_loader, teacher, student, model_save_path1)
        record = {
            'training_loss1': training_loss_list1,
            'testing_accuracy1': testing_accuracy_list1
        }

        # create stats directories
        dirs = os.path.dirname(statistics_save_path1)
        os.makedirs(dirs, exist_ok=True)
        if args.n_training_epochs1 > 0 and (not args.flag_debug):
            torch.save(record, statistics_save_path1)
        print('===== training stage 1 finish. =====')

        # load best model found in stage 1
        if not args.flag_debug:
            record = torch.load(model_save_path1, map_location='cpu')
            best_testing_accuracy = record['testing_accuracy']
            student.load_state_dict(record['state_dict'])
            print('===== best model in stage 1 loaded, testing acc = %f. =====' % (record['testing_accuracy']))

    # model save path and statistics save path for stage 2
    model_save_path2 = 'saves/trained_students/' + \
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
    statistics_save_path2 = 'saves/student_statistics/' + \
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
                            '.stat'

    # model training stage 2
    training_loss_list2, teaching_loss_list2, training_accuracy_list2, testing_accuracy_list2 = \
        train_stage2(args, train_data_loader, test_data_loader, teacher, student, model_save_path2)
    record = {
        'training_loss2': training_loss_list2,
        'teaching_loss2': teaching_loss_list2,
        'training_accuracy2': training_accuracy_list2,
        'testing_accuracy2': testing_accuracy_list2
    }

    # create stats directories
    dirs = os.path.dirname(statistics_save_path2)
    os.makedirs(dirs, exist_ok=True)
    if args.n_training_epochs2 > 0 and (not args.flag_debug):
        torch.save(record, statistics_save_path2)
    print('===== training stage 2 finish. =====')

    # load best model found in stage 2
    if not args.flag_debug:
        record = torch.load(model_save_path2)
        best_testing_accuracy = record['testing_accuracy']
        student.load_state_dict(record['state_dict'])
        print('===== best model in stage 2 loaded, testing acc = %f. =====' % (record['testing_accuracy']))
    
    display_args(args)