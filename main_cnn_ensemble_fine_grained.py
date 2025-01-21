import argparse
import yaml

import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torch.distributed as dist

from torchvision.transforms import Compose, Resize
from models.ResNet_Ensemble import *
from models.utils import *

from models.losses import Ensemble_CE_Loss
from torchvision.models import resnet50
from typing import OrderedDict


import importlib
from tqdm import tqdm

from datasets.utils import get_dataloader
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
cudnn.benchmark = True


def evaluate(classifier, test_loader, device='cuda'):
    torch.cuda.empty_cache()
    classifier.eval()
    total_preds, total_labels, total_preds_last, total_preds_first, total_preds_second = [], [], [], [], []

    for data in tqdm(test_loader):
        real_x = data[0].to(device)
        real_y = data[1].to(device)

        with torch.no_grad():
            logits, _ = classifier(real_x)

            logits_ensemble = 0
            
            for i in range(0, len(logits)):
                logits_ensemble += torch.softmax(logits[i], dim=1)

            _, pred = torch.max(logits_ensemble, dim=1)
            total_preds.append(torch2numpy(pred))
            total_labels.append(torch2numpy(real_y))

            _, pred_last = torch.max(logits[-1], dim=1)
            total_preds_last.append(torch2numpy(pred_last))

            _, pred_first = torch.max(logits[0], dim=1)
            total_preds_first.append(torch2numpy(pred_first))

            _, pred_second = torch.max(logits[1], dim=1)
            total_preds_second.append(torch2numpy(pred_second))

    total_preds = np.concatenate(total_preds, axis=0)
    total_labels= np.concatenate(total_labels, axis=0)
    total_preds_last = np.concatenate(total_preds_last, axis=0)
    total_preds_first = np.concatenate(total_preds_first, axis=0)
    total_preds_second = np.concatenate(total_preds_second, axis=0)

    test_acc = float(np.sum(total_preds==total_labels))/total_preds.shape[0]
    test_acc_last = float(np.sum(total_preds_last==total_labels))/total_preds_last.shape[0]
    test_acc_first = float(np.sum(total_preds_first==total_labels))/total_preds_first.shape[0]
    test_acc_second = float(np.sum(total_preds_second==total_labels))/total_preds_second.shape[0]
    
    return test_acc, test_acc_last, test_acc_first, test_acc_second
    


def train(config):
    training_opt = config['training_opt']
    dataset_cofig = config['dataset']
    networks_args = config['networks']
    optimizer_args = config['optim_params']
    rank = config['local_rank']
    #print('rank=', rank)

    #step 1: prepare data_loader
    if rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_loader = get_dataloader(options=dataset_cofig)
    else:
        #device = torch.device('cuda', config['local_rank'])
        # Setup DDP:
        torch.cuda.set_device(rank)
        #device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend='nccl', rank=rank)
        #dist.init_process_group("nccl")
        #device = rank % torch.cuda.device_count()
        device = torch.device('cuda', config['local_rank'])
        #torch.cuda.set_device(rank)
        data_loader = get_dataloader(distributed=True, options=dataset_cofig)


    log_dir = str(training_opt['log_dir']) + str(networks_args['params']['depth'])
    log_dir = os.path.join(log_dir, str(training_opt['mixer_type']))
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'log.txt')

    #step 2: prepare CNN or ViT
    def_file = networks_args['def_file']
    model_args = networks_args['params']
    # model_args.update({'test': self.test_mode})

    classifier = source_import(def_file).create_model(**model_args)
    if training_opt['pretrained']:
        official_model = resnet50(pretrained=True)
        checkpoint_dict = OrderedDict()
        for key in official_model.state_dict():
            if not key.startswith("fc"):
                checkpoint_dict[key] = official_model.state_dict()[key]
                
        classifier.load_state_dict(checkpoint_dict, strict=False)

    if rank==-1:
        classifier = nn.DataParallel(classifier)
        classifier = classifier.to(device)
    else:
        classifier = nn.parallel.DistributedDataParallel(classifier.to(device), broadcast_buffers=False, device_ids=[rank], output_device=rank)
        #classifier = classifier.to(device)

    #step 3: data mixer
    if training_opt['mixer_type']=='mixup':
        data_mixer = mixup_data
    elif training_opt['mixer_type']=='cutmix':
        data_mixer = cutmix

    #step 4: optimizer
    optimizer_args = config['optim_params']
    optimizer = optim.SGD(classifier.parameters(), lr=optimizer_args['lr'], 
                          momentum=optimizer_args['momentum'], 
                          weight_decay=optimizer_args['weight_decay'])
    
    #scheduler
    if config['coslr']:
        print("===> Using coslr eta_min={}".format(config['endlr']))
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=training_opt['warmup_epoch'], t_total=training_opt['num_epochs'])

        
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=training_opt['milestones'], gamma=0.1)
        

    acc_record = 0
    total_steps = 0
    for epoch in range(1, training_opt['num_epochs'] + 1):
        torch.cuda.empty_cache()
        total_preds = []
        total_labels = []
        total_loss = 0

        classifier.train()
        #scheduler
        scheduler.step()

        if rank!=-1:
            data_loader['train'].sampler.set_epoch(epoch)

        for data in tqdm(data_loader['train']):
            total_steps += 1

            real_x = data[0].to(device)
            real_y = data[1].to(device)
            mixed_x, y_a, y_b, lam, mixed_index = data_mixer(real_x, real_y, alpha=training_opt['alpha'])
            logits_mixed, features_mixed = classifier(mixed_x)
            logits, features = classifier(real_x)
            #logit_scale = classifier.log_scale.exp()
            weights = None
            loss = 0
            logits_ensemble = 0
            for i in range(0, len(logits)):
                loss_i, weights = Ensemble_CE_Loss(logits=logits[i], logits_mixed=logits_mixed[i], features=features[i], 
                                           features_mixed=features_mixed[i], y_a=y_a, y_b=y_b, weights=weights,
                                           mixed_loss=training_opt['mixed_loss'], 
                                           base_weight=training_opt['base_weight'], gamma=training_opt['gamma'], 
                                           lam=lam)
                logits_ensemble += torch.softmax(logits[i], dim=1)
                loss = loss + loss_i
            
            loss.backward()
            total_loss = total_loss + loss.item()

            _, pred = torch.max(logits_ensemble, dim=1)
            total_preds.append(torch2numpy(pred))
            total_labels.append(torch2numpy(real_y))

            if total_steps % training_opt['num_accmutations']==0 or total_steps % len(data_loader['train'])==0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

        total_loss /= len(data_loader['train'])
        total_preds = np.concatenate(total_preds, axis=0)
        total_labels= np.concatenate(total_labels, axis=0)
        train_acc = float(np.sum(total_preds==total_labels))/total_preds.shape[0]

        test_acc_ensemble, test_acc_last, test_acc_first, test_acc_second = evaluate(classifier, data_loader['test'])

        lr_current = max([param_group['lr'] for param_group in optimizer.param_groups])

        log_str = ['[%d/%d]  learing_rate: %.5f  loss: %.4f  train_acc: %.4f  test_acc_ensemble: %.4f   test_acc_last: %.4f   test_acc_first: %.4f  test_acc_second: %.4f' 
                   % (epoch, training_opt['num_epochs'], lr_current, total_loss, train_acc, test_acc_ensemble, test_acc_last, test_acc_first, test_acc_second)]
        
        
        #print(log_str)
        print_write(log_str, log_file)
    
        states = {
            'epoch':epoch,
            'acc': train_acc,
            'test_acc': test_acc_ensemble,
            'classifier_dict':classifier.state_dict(),
            'optimizer':optimizer.state_dict()
        }

        if train_acc>acc_record:
            acc_record = train_acc
            filename = "best_checkpoint_%d.pth" % (rank)
            filename = os.path.join(log_dir, filename)
            save_checkpoint(states, filename)
        
        filename = "last_checkpoint_%d.pth" % (rank)
        filename = os.path.join(log_dir, filename)
        save_checkpoint(states, filename)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./config/fgvc/CUB_200_2011/resnet_ensemble.yaml', type=str)
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)

    args = parser.parse_args()

    depths = [50]
    
    mixer_types = ['mixup']
    for mixer_type in mixer_types:    #testing with different mixup methods
        for depth in depths:   #testing with different depth 
            with open(args.cfg) as f:
                config = yaml.safe_load(f)
            '''
            if depth!=18:
                config['optim_params']['lr'] = 0.05
            '''
            config['training_opt']['mixer_type'] = mixer_type
            config['networks']['params']['depth'] = depth
            config['local_rank'] = args.local_rank
        
            train(config)
            
    print('ALL COMPLETED.')
