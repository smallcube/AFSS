import os
import torch
import dateutil.tz
from datetime import datetime
import time
import logging
import numpy as np
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_loader.dataloader import CIFAR10, CIFAR100, ImageNet, CUB_200_2011, TinyImageNet, Standford_Cars


def get_dataloader(distributed=False, options=None):
    print("{} Preparation".format(options['dataset']))
    if 'cifar10' == options['dataset']:
        Data = CIFAR10(distributed=distributed, **options)
        trainloader, testloader = Data.trainloader, Data.testloader
    elif 'cifar100' in options['dataset']:
        Data = CIFAR100(distributed=distributed, **options)
        trainloader, testloader = Data.trainloader, Data.testloader
    elif 'ImageNet' in options['dataset']:
        Data = ImageNet(distributed=distributed, **options)
        trainloader, testloader = Data.trainloader, Data.testloader
    elif 'CUB_200_2011' in options['dataset']:
        Data = CUB_200_2011(distributed=distributed, **options)
        trainloader, testloader = Data.trainloader, Data.testloader
    elif 'cars' in options['dataset']:
        Data = Standford_Cars(distributed=distributed, **options)
        trainloader, testloader = Data.trainloader, Data.testloader
    elif 'tiny-imagenet-200' in options['dataset']:
        Data = TinyImageNet(distributed=distributed, **options)
        trainloader, testloader = Data.trainloader, Data.testloader

        
    dataloader = {'train': trainloader, 'test': testloader}
    return dataloader

def get_dataloader2(options):
    print("{} Preparation".format(options['dataset']))
    if 'cifar10' == options['dataset']:
        Data = CIFAR10(**options)
        trainloader, testloader = Data.trainloader, Data.testloader
    elif 'cifar100' in options['dataset']:
        Data = CIFAR100(**options)
        trainloader, testloader = Data.trainloader, Data.testloader
    elif 'ImageNet' in options['dataset']:
        Data = ImageNet(**options)
        trainloader, testloader = Data.trainloader, Data.testloader
    elif 'ImageNet_MoCo' in options['dataset']:
        Data = ImageNet_MoCo(**options)
        trainloader, testloader = Data.trainloader, Data.testloader
    elif 'CUB_200_2011' in options['dataset']:
        Data = CUB_200_2011(**options)
        trainloader, testloader = Data.trainloader, Data.testloader
        
    dataloader = {'train': trainloader, 'test': testloader}
    return dataloader