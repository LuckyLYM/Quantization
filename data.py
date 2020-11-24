import os
import torch
import pickle
import numpy
import torchvision.transforms as transforms

def _data_transforms_cifar10():
  # data statistics
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])


  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def _data_transforms_SVHN():
  # data statistics
  SVHN_MEAN = [0.5, 0.5, 0.5]
  SVHN_STD = [0.5, 0.5, 0.5]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
  ])


  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])
  return train_transform, valid_transform


class dataset():
    def __init__(self, root=None, train=True):
        self.root = root
        self.train = train
        self.transform = transforms.ToTensor()
        if self.train:
            train_data_path = os.path.join(root, 'train_data')
            train_labels_path = os.path.join(root, 'train_labels')
            self.train_data = numpy.load(open(train_data_path, 'r'))
            self.train_data = torch.from_numpy(self.train_data.astype('float32'))
            self.train_labels = numpy.load(open(train_labels_path, 'r')).astype('int')
        else:
            test_data_path = os.path.join(root, 'test_data')
            test_labels_path = os.path.join(root, 'test_labels')
            self.test_data = numpy.load(open(test_data_path, 'r'))
            self.test_data = torch.from_numpy(self.test_data.astype('float32'))
            self.test_labels = numpy.load(open(test_labels_path, 'r')).astype('int')

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]


        return img, target
