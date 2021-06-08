# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2021-06-08 16:16:42
"""

import os
import pickle

def check_dataset(dataset_path):
    train_data_file_path = dataset_path + 'train'
    test_data_file_path = dataset_path + 'test'
    with open(train_data_file_path, 'rb') as fp:
        train_data = pickle.load(fp)
    print(train_data)


if __name__ == '__main__':
    dataset_path = '../datasets/CIFAR-100/'
    check_dataset(dataset_path)