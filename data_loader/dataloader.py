import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.nn import functional as F
import torchvision.transforms as transforms
from PIL import Image
import torchvision.datasets as datasets

import numpy as np
import pandas as pd
import data_loader.moco_loader as moco_loader

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive, check_integrity, download_url, verify_str_arg

from data_loader.autoaugment import CIFAR10Policy, ImageNetPolicy


class CIFAR10(object):
    def __init__(self, distributed=False, **options):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616])
        
        
        if options['img_size']>=40:
            if "autoaugment" in options and options['autoaugment']==True:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(options['img_size']),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(options['img_size']),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
        
        else:
            if "autoaugment" in options and options['autoaugment']==True:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(options['img_size'], padding=4, fill=128),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(options['img_size'], padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
        
        transform = transforms.Compose([
            transforms.Resize(options['img_size']),
            transforms.ToTensor(),
            normalize,
        ])
        
        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'cifar10')

        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        
        if distributed:
            sampler_train = DistributedSampler(trainset)
            trainloader = DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_train,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=True
            )
            
            sampler_test = DistributedSampler(testset)
            testloader = DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_test,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=False
            )

        else:
            trainloader = DataLoader(
                trainset, 
                batch_size=batch_size, shuffle=True,
                num_workers=options['num_workers'], pin_memory=pin_memory,
            )
            
            
            testloader = DataLoader(
                testset, 
                batch_size=batch_size, shuffle=False,
                num_workers=options['num_workers'], pin_memory=pin_memory,
            )
        
        

        self.num_classes = 10
        self.trainloader = trainloader
        self.testloader = testloader

class CIFAR100(object):
    def __init__(self, distributed=False, **options):
        '''
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010])
        '''
        '''
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
       
        '''
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        
        
        
        if options['img_size']>=40:
            
            if "autoaugment" in options and options['autoaugment']==True:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(options['img_size']),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(options['img_size']),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
        
        else:
            if "autoaugment" in options and options['autoaugment']==True:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(options['img_size'], padding=4, fill=128),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(options['img_size'], padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
        
        transform = transforms.Compose([
            transforms.Resize(options['img_size']),
            transforms.ToTensor(),
            normalize,
        ])
        

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'cifar100')

        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
        
        if distributed:
            sampler_train = DistributedSampler(trainset)
            trainloader = DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_train,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=True
            )
            
            sampler_test = DistributedSampler(testset)
            testloader = DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_test,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=False
            )

        else:
            trainloader = DataLoader(
                trainset, 
                batch_size=batch_size, shuffle=True,
                num_workers=options['num_workers'], pin_memory=pin_memory,
            )
            
            
            testloader = DataLoader(
                testset, 
                batch_size=batch_size, shuffle=False,
                num_workers=options['num_workers'], pin_memory=pin_memory,
            )
        
        self.num_classes = 100
        self.trainloader = trainloader
        self.testloader = testloader



class ImageNet_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        file_name = os.path.join(root, txt)
        with open(file_name) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index

class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

class ImageNet(object):
    def __init__(self, distributed=False, **options):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        
        batch_size = options['batch_size']
        pin_memory = True if options['use_gpu'] else False
        #print("distributed=", distributed)
        if "moco" in options and options["moco"]==True:
            if options['aug_plus']==True:
                #moco v2
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([moco_loader.GaussianBlur([0.1, 2.0])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                #moco v1
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            #print("distributed11111111111")
            trainset = datasets.ImageFolder(os.path.join(options['dataroot'], "train/"), moco_loader.TwoCropsTransform(transform_train))

        else:
            if "autoaugment" in options and options['autoaugment']==True:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(options['img_size']),
                    transforms.RandomHorizontalFlip(),
                    ImageNetPolicy(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(options['img_size']),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            trainset = ImageNet_Dataset(root=options['dataroot'], txt='ImageNet_train.txt', transform=transform_train)

        transform_test = transforms.Compose([
            transforms.Resize(options['img_size']+32),
            transforms.CenterCrop(options['img_size']),
            transforms.ToTensor(),
            normalize,
        ])

        
        testset = ImageNet_Dataset(root=options['dataroot'], txt='ImageNet_val.txt', transform=transform_test)

        if distributed:
            #print("distributed")
            sampler_train = DistributedSampler(trainset)
            trainloader = DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_train,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=True
            )
            
            sampler_test = DistributedSampler(testset)
            testloader = DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_test,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=False
            )

        else:
            trainloader = DataLoader(
                trainset, 
                batch_size=batch_size, shuffle=True,
                num_workers=options['num_workers'], pin_memory=pin_memory,
            )
            
            
            testloader = DataLoader(
                testset, 
                batch_size=batch_size, shuffle=False,
                num_workers=options['num_workers'], pin_memory=pin_memory,
            )
            
        self.num_classes = 1000
        self.trainloader = trainloader
        self.testloader = testloader



class CUB_200_2011(object):
    def __init__(self, distributed=False, **options):
        '''
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
        '''
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        
        if options['img_size']>=40:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop((options['img_size'], options['img_size'])),
                transforms.RandomCrop(options['img_size']-64, padding=8),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(15),
                transforms.ToTensor(),
                #normalize,
            ])
        
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(options['img_size'], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize,
            ])

        transform = transforms.Compose([
            transforms.Resize((options['img_size'], options['img_size'])),
            transforms.CenterCrop(options['img_size']-64),
            transforms.ToTensor(),
            #normalize,
        ])
        

        batch_size = options['batch_size']
        #data_root = os.path.join(options['dataroot'], 'CUB_200_2011')

        pin_memory = True if options['use_gpu'] else False

        trainset = Cub2011(options['dataroot'], train=True, transform=transform_train, download=False)
        testset = Cub2011(options['dataroot'], train=False, transform=transform, download=False)

        if distributed:
            sampler_train = DistributedSampler(trainset)
            trainloader = DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_train,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=True
            )
            
            sampler_test = DistributedSampler(testset)
            testloader = DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_test,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=False
            )

        else:
            trainloader = DataLoader(
                trainset, 
                batch_size=batch_size, shuffle=True,
                num_workers=options['num_workers'], pin_memory=pin_memory,
            )
            
            
            testloader = DataLoader(
                testset, 
                batch_size=batch_size, shuffle=False,
                num_workers=options['num_workers'], pin_memory=pin_memory,
            )
        

        self.num_classes = 200
        self.trainloader = trainloader
        self.testloader = testloader

class Standford_Cars(object):
    def __init__(self, distributed=False, **options):
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                        std=[0.2675, 0.2565, 0.2761])
        
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(options['img_size']),
            transforms.RandomCrop(options['img_size']-64, padding=8),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(15),
            transforms.ToTensor(),
            #normalize,
        ])
        
        
        transform = transforms.Compose([
            transforms.Resize(options['img_size']),
            transforms.CenterCrop(options['img_size']-64),
            transforms.ToTensor(),
            #normalize,
        ])
        

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'])

        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.StanfordCars(root=data_root, split="train", download=True, transform=transform_train)
        testset = torchvision.datasets.StanfordCars(root=data_root, split="test", download=True, transform=transform)
        
        if distributed:
            sampler_train = DistributedSampler(trainset)
            trainloader = DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_train,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=True
            )
            
            sampler_test = DistributedSampler(testset)
            testloader = DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_test,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=False
            )

        else:
            trainloader = DataLoader(
                trainset, 
                batch_size=batch_size, shuffle=True,
                num_workers=options['num_workers'], pin_memory=pin_memory,
            )
            
            
            testloader = DataLoader(
                testset, 
                batch_size=batch_size, shuffle=False,
                num_workers=options['num_workers'], pin_memory=pin_memory,
            )
        
        self.num_classes = 196
        self.trainloader = trainloader
        self.testloader = testloader


class TinyImageNet_Dataset(VisionDataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'tiny-imagenet-200/'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(TinyImageNet_Dataset, self).__init__(root, transform=transform, target_transform=target_transform)

        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))

        if self._check_integrity():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        if not os.path.isdir(self.dataset_path):
            print('Extracting...')
            extract_archive(os.path.join(root, self.filename))

        _, class_to_idx = find_classes(os.path.join(self.dataset_path, 'wnids.txt'))

        self.data = make_dataset(self.root, self.base_folder, self.split, class_to_idx)

    def _download(self):
        print('Downloading...')
        download_url(self.url, root=self.root, filename=self.filename)
        print('Extracting...')
        extract_archive(os.path.join(self.root, self.filename))

    def _check_integrity(self):
        return check_integrity(os.path.join(self.root, self.filename), self.md5)

    def __getitem__(self, index):
        img_path, target = self.data[index]
        image = self.loader(img_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.data)


def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s: s.strip(), r.readlines()))

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


def make_dataset(root, base_folder, dirname, class_to_idx):
    images = []
    dir_path = os.path.join(root, base_folder, dirname)

    if dirname == 'train':
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                for imgname in sorted(os.listdir(cls_imgs_path)):
                    path = os.path.join(cls_imgs_path, imgname)
                    item = (path, class_to_idx[fname])
                    images.append(item)
    else:
        imgs_path = os.path.join(dir_path, 'images')
        imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}

        for imgname in sorted(os.listdir(imgs_path)):
            path = os.path.join(imgs_path, imgname)
            item = (path, class_to_idx[cls_map[imgname]])
            images.append(item)

    return images

class TinyImageNet(object):
    def __init__(self, distributed=False, **options):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        
        batch_size = options['batch_size']
        pin_memory = True if options['use_gpu'] else False
        #print("distributed=", distributed)
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(options['img_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(options['img_size']),
            #transforms.CenterCrop(options['img_size']),
            transforms.ToTensor(),
            normalize,
        ])

        trainset = TinyImageNet_Dataset(root=options['dataroot'], split='train', transform=transform_train, download=True)
        testset = TinyImageNet_Dataset(root=options['dataroot'], split='val', transform=transform_test, download=True)
        
        if distributed:
            #print("distributed")
            sampler_train = DistributedSampler(trainset)
            trainloader = DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_train,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=True
            )
            
            sampler_test = DistributedSampler(testset)
            testloader = DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_test,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=False
            )

        else:
            trainloader = DataLoader(
                trainset, 
                batch_size=batch_size, shuffle=True,
                num_workers=options['num_workers'], pin_memory=pin_memory,
            )
            
            
            testloader = DataLoader(
                testset, 
                batch_size=batch_size, shuffle=False,
                num_workers=options['num_workers'], pin_memory=pin_memory,
            )
            
        self.num_classes = 200
        self.trainloader = trainloader
        self.testloader = testloader