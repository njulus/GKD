# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2021-06-08 16:16:42
"""

import pickle
import numpy as np

def generate_label_q_cifar():
    path = 'D:/Experiment Datasets/Image Datasets/CIFAR-FS/'
    
    train_data_path = path + 'CIFAR_FS_train.pickle'
    with open(train_data_path, 'rb') as fp:
        train_data = pickle.load(fp, encoding='bytes')
    train_labels = train_data[b'labels']
    train_labels = np.unique(np.array(train_labels))
    
    val_data_path = path + 'CIFAR_FS_val.pickle'
    with open(val_data_path, 'rb') as fp:
        val_data = pickle.load(fp, encoding='bytes')
    val_labels = val_data[b'labels']
    val_labels = np.unique(np.array(val_labels))

    test_data_path = path + 'CIFAR_FS_test.pickle'
    with open(test_data_path, 'rb') as fp:
        test_data = pickle.load(fp, encoding='bytes')
    test_labels = test_data[b'labels']
    test_labels = np.unique(np.array(test_labels))

    label_q = np.concatenate([train_labels, val_labels, test_labels])
    # print(label_q)
    np.save('../datasets/CIFAR-100/label_q.npy', label_q)



def generate_label_q_cub():
    dataset_path = '../datasets/CUB-200/'
    train_data_file_path = dataset_path + 'train'
    with open(train_data_file_path, 'rb') as fp:
        train_data = pickle.load(fp, encoding='bytes')
    labels = train_data['labels']
    
    labels = np.array(labels)
    n_classes = np.max(labels) + 1    
    label_q = np.arange(n_classes)
    print(label_q)
    np.save(dataset_path + 'label_q.npy', label_q)



if __name__ == '__main__':
    generate_label_q_cifar()
    generate_label_q_cub()