# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2020-12-08 20:59:35
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.nn import functional as F

from pytorch_metric_learning.miners import TripletMarginMiner

from Test import test, test_ncm
from utils.triplet import merge

def pretrain(args, train_data_loader, test_data_loader, network, model_save_path):
    loss_function = nn.CrossEntropyLoss()
    optimizer = SGD(params=network.parameters(), lr=args.lr, weight_decay=args.wd,
                    momentum=args.mo, nesterov=True)
    if args.gamma != -1:
        scheduler = MultiStepLR(optimizer, args.point, args.gamma)
    else:
        scheduler = CosineAnnealingLR(optimizer, args.n_training_epochs, 0.001 * args.lr)

    training_loss_list = []
    training_accuracy_list = []
    testing_accuracy_list = []
    best_testing_accuracy = 0

    for epoch in range(1, args.n_training_epochs + 1):
        training_loss = 0
        training_accuracy = 0

        network.train()
        for batch_index, batch in enumerate(train_data_loader):
            images, labels = batch
            images = images.float().cuda(args.devices[0])
            labels = labels.long().cuda(args.devices[0])

            logits = network.forward(images)
            loss_value = loss_function(logits, labels)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            prediction = torch.argmax(logits, dim=1)
            training_loss += loss_value.cpu().item() * images.size()[0]
            training_accuracy += torch.sum((prediction == labels).float()).cpu().item()

        training_loss /= train_data_loader.dataset.__len__()
        training_loss_list.append(training_loss)
        training_accuracy /= train_data_loader.dataset.__len__()
        training_accuracy_list.append(training_accuracy)
        testing_accuracy = test(args, test_data_loader, network)
        testing_accuracy_list.append(testing_accuracy)

        print('epoch %d finish: training_loss = %f, training_accuracy = %f, testing_accuracy = %f' % (
            epoch, training_loss, training_accuracy, testing_accuracy
        ))

        # if we find a better model
        if not args.flag_debug:
            if testing_accuracy > best_testing_accuracy:
                best_testing_accuracy = testing_accuracy
                record = {
                    'state_dict': network.state_dict(),
                    'testing_accuracy': testing_accuracy,
                    'epoch': epoch
                }
                torch.save(record, model_save_path)

        scheduler.step()

    return training_loss_list, training_accuracy_list, testing_accuracy_list



def train_stage1(args, train_data_loader, test_data_loader, teacher, student, model_save_path1):
    loss_function = nn.KLDivLoss(reduction='none')
    optimizer = SGD(params=student.parameters(), lr=args.lr1, weight_decay=args.wd,
        momentum=args.mo, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, args.n_training_epochs1, 0.1 * args.lr1)
    miner = TripletMarginMiner(margin=0.2, type_of_triplets='semihard')

    training_loss_list1 = []
    testing_accuracy_list1 = []
    best_testing_accuracy = 0

    for epoch in range(1, args.n_training_epochs1 + 1):
        training_loss = 0
        n_tuples = 0

        student.train()
        for batch_index, batch in enumerate(train_data_loader):
            images, labels = batch
            images = images.float().cuda(args.devices[0])
            labels = labels.long().cuda(args.devices[0])

            with torch.no_grad():
                teacher_embeddings = teacher.forward(images, flag_embedding=True)
                teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=1)
            
            student_embeddings = student.forward(images, flag_embedding=True)
            student_embeddings = F.normalize(student_embeddings, p=2, dim=1)

            with torch.no_grad():
                anchor_id, positive_id, negative_id = miner(student_embeddings, labels)
                merged_anchor_id, merged_positive_id, merged_negative_id = \
                    merge(args, anchor_id, positive_id, negative_id)

            with torch.no_grad():
                teacher_anchor = teacher_embeddings[merged_anchor_id]
                teacher_positive = teacher_embeddings[merged_positive_id]
                teacher_negative = teacher_embeddings[merged_negative_id]

                teacher_ap_dist = torch.norm(teacher_anchor - teacher_positive, p=2, dim=1)
                teacher_an_dist = torch.norm(teacher_anchor.unsqueeze(1) - teacher_negative, p=2, dim=2)

                teacher_tuple_logits = torch.cat([-teacher_ap_dist.unsqueeze(1), -teacher_an_dist], dim=1) / args.tau1

            student_anchor = student_embeddings[merged_anchor_id]
            student_positive = student_embeddings[merged_positive_id]
            student_negative = student_embeddings[merged_negative_id]

            student_ap_dist = torch.norm(student_anchor - student_positive, p=2, dim=1)
            student_an_dist = torch.norm(student_anchor.unsqueeze(1) - student_negative, p=2, dim=2)

            student_tuple_logits = torch.cat([-student_ap_dist.unsqueeze(1), -student_an_dist], dim=1) / args.tau1

            loss_value = 1000 * loss_function(F.log_softmax(student_tuple_logits), F.softmax(teacher_tuple_logits))
            loss_value = torch.mean(torch.sum(loss_value, dim=1))
        
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            training_loss += loss_value.cpu().item() * student_tuple_logits.size()[0]
            n_tuples += student_tuple_logits.size()[0]
        
        training_loss /= n_tuples
        training_loss_list1.append(training_loss)

        if epoch % 10 == 0:
            testing_accuracy = test_ncm(args, train_data_loader, test_data_loader, student)
            testing_accuracy_list1.append(testing_accuracy)

            print('epoch %d finish: training_loss = %f, testing_accuracy = %f' % (epoch, training_loss, testing_accuracy))

            if not args.flag_debug:
                if testing_accuracy > best_testing_accuracy:
                    best_testing_accuracy = testing_accuracy
                    record = {
                        'state_dict': student.state_dict(),
                        'testing_accuracy': testing_accuracy,
                        'epoch': epoch
                    }
                    torch.save(record, model_save_path1)
        else:
            print('epoch %d finish: training_loss = %f' % (epoch, training_loss))
    
        scheduler.step()

    return training_loss_list1, testing_accuracy_list1



def train_stage2(args, train_data_loader, test_data_loader, teacher, student, model_save_path2):
    training_loss_function = nn.CrossEntropyLoss()
    teaching_loss_function = nn.KLDivLoss(reduction='batchmean')
    optimizer = SGD(params=student.parameters(), lr=args.lr2, weight_decay=args.wd,
        momentum=args.mo, nesterov=True)
    if args.gamma != -1:
        scheduler = MultiStepLR(optimizer, args.point, args.gamma)
    else:
        scheduler = CosineAnnealingLR(optimizer, args.n_training_epochs2, 0.001 * args.lr2)

    if args.model_name == 'gkd':
        class_center_file_path = 'saves/class_centers/' + args.data_name + '_' + args.teacher_network_name + \
            '_class=' + str(args.n_classes) + \
            '_newclass=' + str(args.n_new_classes) + \
            '.center'
        if os.path.exists(class_center_file_path):
            class_centers = torch.load(class_center_file_path)
            class_centers = class_centers.cuda(args.devices[0])
        else:
            class_centers = torch.zeros((args.n_classes, teacher.fc.in_features)).cuda(args.devices[0])
            class_count = torch.zeros(args.n_classes).cuda(args.devices[0])
            for batch_index, batch in enumerate(train_data_loader):
                images, labels = batch
                images = images.float().cuda(args.devices[0])
                labels = labels.long().cuda(args.devices[0])
                
                with torch.no_grad():
                    teacher_embeddings = teacher.forward(images, flag_embedding=True)
                    for i in range(0, args.n_classes):
                        index_of_class_i = (labels == i)
                        class_centers[i] += torch.sum(teacher_embeddings[index_of_class_i], dim=0)
                        class_count[i] += index_of_class_i.size()[0]
            class_count = class_count.unsqueeze(1)
            class_centers = class_centers / class_count
            class_centers = F.normalize(class_centers, p=2, dim=1)
            torch.save(class_centers, class_center_file_path)
        print('===== teacher class centers ready. =====')

    training_loss_list2 = []
    teaching_loss_list2 = []
    training_accuracy_list2 = []
    testing_accuracy_list2 = []
    best_testing_accuracy = 0

    for epoch in range(1, args.n_training_epochs2 + 1):
        training_loss = 0
        teaching_loss = 0
        training_accuracy = 0

        student.train()
        for batch_index, batch in enumerate(train_data_loader):
            images, labels = batch
            images = images.float().cuda(args.devices[0])
            labels = labels.long().cuda(args.devices[0])

            logits = student.forward(images)
            training_loss_value = training_loss_function(logits, labels)

            if args.model_name == 'ce':
                total_loss_value = training_loss_value
            elif args.model_name == 'kd':
                with torch.no_grad():
                    teacher_logits = teacher.forward(images)
                teaching_loss_value = args.lambd * teaching_loss_function(
                    F.log_softmax(logits / args.tau2, dim=1),
                    F.softmax(teacher_logits / args.tau2, dim=1)
                )
                total_loss_value = training_loss_value + teaching_loss_value
            elif args.model_name == 'gkd':
                with torch.no_grad():
                    label_table = torch.arange(args.n_classes).long().unsqueeze(1).cuda(args.devices[0])
                    class_in_batch = (labels == label_table).any(dim=1)
                    class_centers_in_batch = class_centers[class_in_batch]

                    teacher_embeddings = teacher.forward(images, flag_embedding=True)
                    teacher_logits = torch.mm(teacher_embeddings, class_centers_in_batch.t())
                teaching_loss_value = args.lambd * teaching_loss_function(
                    F.log_softmax(logits[:, class_in_batch] / args.tau2, dim=1),
                    F.softmax(teacher_logits / args.tau2, dim=1)
                )
                total_loss_value = training_loss_value + teaching_loss_value
            
            optimizer.zero_grad()
            total_loss_value.backward()
            optimizer.step()

            prediction = torch.argmax(logits, dim=1)
            training_loss += training_loss_value.cpu().item() * images.size()[0]
            if args.model_name in {'kd', 'gkd'}:
                teaching_loss += teaching_loss_value.cpu().item() * images.size()[0]
            else:
                pass
            training_accuracy += torch.sum((prediction == labels).float()).cpu().item()
        
        training_loss /= train_data_loader.dataset.__len__()
        training_loss_list2.append(training_loss)
        teaching_loss /= train_data_loader.dataset.__len__()
        teaching_loss_list2.append(teaching_loss)
        training_accuracy /= train_data_loader.dataset.__len__()
        training_accuracy_list2.append(training_accuracy)
        testing_accuracy = test(args, test_data_loader, student)
        testing_accuracy_list2.append(testing_accuracy)

        print('epoch %d finish: training_loss = %f, teaching_loss = %f, training_accuracy = %f, testing_accuracy = %f' % (
            epoch, training_loss, teaching_loss, training_accuracy, testing_accuracy
        ))

        # if we find a better model
        if not args.flag_debug:
            if testing_accuracy > best_testing_accuracy:
                best_testing_accuracy = testing_accuracy
                record = {
                    'state_dict': student.state_dict(),
                    'testing_accuracy': testing_accuracy,
                    'epoch': epoch
                }
                torch.save(record, model_save_path2)

        scheduler.step()

    return training_loss_list2, teaching_loss_list2, training_accuracy_list2, testing_accuracy_list2