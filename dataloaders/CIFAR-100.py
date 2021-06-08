# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2020-12-08 15:42:03
"""

import pickle
from PIL import Image

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, data_path, flag_mode, n_new_classes):
        super(MyDataset, self).__init__()
        self.data_path = data_path
        self.flag_mode = flag_mode
        self.n_new_classes = n_new_classes

        self.features, self.labels = self.read_data()

        self.transform_augment = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])
        self.transform_simple = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])
    
    def read_data(self):
        if self.flag_mode == 'train':
            data_file_path = self.data_path + 'train'
        elif self.flag_mode == 'test':
            data_file_path = self.data_path + 'test'

        with open(data_file_path, 'rb') as fp:
            data = pickle.load(fp, encoding='bytes')
        features = np.array(data[b'data'])
        labels = np.array(data[b'fine_labels'])

        self.label_q = np.load(self.data_path + 'label_q.npy')
        mapping = {label:pos for pos, label in enumerate(self.label_q)}
        positions = np.array([mapping[label] for label in labels])
        indexes_needed = np.argwhere((positions >= self.n_new_classes) & (positions < 50 + self.n_new_classes)).squeeze(1)

        features_needed = features[indexes_needed]
        labels_needed = labels[indexes_needed]

        self.label2y = {}
        current_y = 0
        label_table = np.sort(np.unique(labels_needed))
        for label in label_table:
            if label in self.label2y.keys():
                continue
            else:
                self.label2y[label] = current_y
                current_y += 1

        return features_needed, labels_needed

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        feature = self.features[index, :]
        # reshape feature to the shape of (height, width, depth)
        feature_r = feature[:1024].reshape(32, 32)
        feature_g = feature[1024:2048].reshape(32, 32)
        feature_b = feature[2048:].reshape(32, 32)
        feature = np.dstack((feature_r, feature_g, feature_b))
        image = Image.fromarray(feature)
        # data preprocess
        if self.flag_mode == 'train':
            image = self.transform_augment(image)
        else:
            image = self.transform_simple(image)
        label = self.labels[index]
        y = self.label2y[label]
        return image, y

    def get_n_classes(self):
        assert(len(np.unique(self.labels)) == 50)
        return 50



def generate_data_loader(data_path, flag_mode, n_new_classes, batch_size, n_workers):
    my_dataset = MyDataset(data_path, flag_mode, n_new_classes)
    my_data_loader = DataLoader(my_dataset, batch_size, shuffle=True, num_workers=n_workers)
    return my_data_loader



if __name__ == '__main__':
    data_path = '../datasets/CIFAR-100/'
    flag_mode = 'train'
    n_new_classes = 0
    batch_size = 2
    n_workers = 0

    my_data_loader = generate_data_loader(data_path, flag_mode, n_new_classes, batch_size, n_workers)
    for batch_index, batch in enumerate(my_data_loader):
        image, label = batch
        print(image.size())
        print(label.size())
        break

    print(my_data_loader.dataset.get_n_classes())