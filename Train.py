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

from Test import test
from utils.triplet import merge

def pretrain(args, train_data_loader, test_data_loader, network, model_save_path):
    loss_function = nn.CrossEntropyLoss()
    optimizer = SGD(params=network.parameters(), lr=args.lr, weight_decay=args.wd,
                    momentum=args.mo, nesterov=True)
    scheduler = MultiStepLR(optimizer, args.point, args.gamma)

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

            print(labels)
            logits = network.forward(images)
            print(logits.size())
            loss_value = loss_function(logits, labels)
            print(loss_value)

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
                    'validating_accuracy': testing_accuracy,
                    'epoch': epoch
                }
                torch.save(record, model_save_path)

        scheduler.step()

    return training_loss_list, training_accuracy_list, testing_accuracy_list