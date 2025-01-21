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
from models.losses import *

import importlib
from tqdm import tqdm

from datasets.utils import get_dataloader

os.environ['CUDA_VISIABLE_DEVICES'] = '0, 1'


def evaluate(classifier, test_loader, device='cuda'):
    classifier.eval()
    total_preds, total_labels, total_preds_last, total_preds_first, total_preds_second = [], [], [], [], []

    for data in tqdm(test_loader):
        real_x = data[0].to(device)
        real_y = data[1].to(device)
        logits, _ = classifier(real_x)

        logits_ensemble = 0
        
        for i in range(len(logits)):
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

    log_dir = str(training_opt['log_dir']) + str(networks_args['params']['model_name'])
    if "pretrained" in networks_args['params'] and networks_args['params']['pretrained']==True:
        log_dir += "_pretrained_"
    
    log_dir = os.path.join(log_dir, str(training_opt['mixer_type']))
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'log.txt')

    print(log_file)

    #step 1: prepare data_loader  & classifier
    def_file = networks_args['def_file']
    model_args = networks_args['params']
    # model_args.update({'test': self.test_mode})

    classifier = source_import(def_file).create_model(**model_args)
    


    if config['local_rank'] != -1:
        torch.cuda.set_device(config['local_rank'])
        #device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend='nccl', rank=config['local_rank'])

        data_loader = get_dataloader(distributed=True, options=dataset_cofig)
        device = torch.device('cuda', config['local_rank'])  

        classifier = classifier.to(device)
        classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[config['local_rank']],
                                                        output_device=config['local_rank'], find_unused_parameters=True)
    else:
        data_loader = get_dataloader(distributed=False, options=dataset_cofig)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

        classifier = classifier.to(device) 
        classifier = nn.DataParallel(classifier)

    #step 2: prepare CNN or ViT
    
    #step 3: data mixer
    if training_opt['mixer_type']=='mixup':
        data_mixer = mixup_data
    elif training_opt['mixer_type']=='cutmix':
        data_mixer = cutmix
    else:
        data_mixer = saliency_mix

    #step 4: optimizer    
    optimizer = optim.SGD(classifier.parameters(), lr=optimizer_args['lr'], 
                          momentum=optimizer_args['momentum'], 
                          weight_decay=optimizer_args['weight_decay'])
    num_steps = training_opt['num_steps']
    #scheduler
    if config['coslr']:
        print("===> Using coslr eta_min={}".format(config['endlr']))
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=training_opt['warmup_steps'], t_total=num_steps)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=training_opt['warmup_steps'], t_total=num_steps)

    acc_record, total_steps, actual_steps = 0, 0, 0
    acc_record = 0
    for epoch in range(1, training_opt['num_epochs'] + 1):
        torch.cuda.empty_cache()
        total_preds = []
        total_labels = []
        total_loss, avg_loss = 0, 0
        count = 0

        if config['local_rank'] != -1:
            data_loader['train'].sampler.set_epoch(epoch)

        classifier.train()

        for data in tqdm(data_loader['train']):
            total_steps = total_steps + 1

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
                loss_i, weights = Ensemble_CE_Loss(logits=logits[i], logits_mixed=logits_mixed[i], 
                                                   features=features[i], features_mixed=features_mixed[i], 
                                                   y_a=y_a, y_b=y_b, 
                                                    gamma=training_opt['gamma'], lam=lam, 
                                                    mixed_loss=training_opt['mixed_loss'], 
                                                    weights=weights, base_weight=training_opt['base_weight'])
                
                logits_ensemble += torch.softmax(logits[i], dim=1)
                loss = loss + loss_i
            
            loss.backward()
            avg_loss = avg_loss + loss
            
            _, pred = torch.max(logits_ensemble, dim=1)
            total_preds.append(torch2numpy(pred))
            total_labels.append(torch2numpy(real_y))

            if total_steps % training_opt['num_accmutations']==0 or total_steps % len(data_loader['train'])==0:
                #print("total_steps=", total_steps, "   num_accmutations=", training_opt['num_accmutations'], "   actual=", actual_steps, "   len=", len(data_loader['train']))
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), training_opt['max_grad_norm'])
                count = count + 1
                actual_steps = actual_steps + 1
                avg_loss = avg_loss / training_opt['num_accmutations']
                total_loss = total_loss + avg_loss.item()
                
                #avg_loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                #avg_loss = 0
                del logits_ensemble
                torch.cuda.empty_cache()
            if actual_steps >= training_opt['num_steps']:
                break
           
        total_loss /= count
        total_preds = np.concatenate(total_preds, axis=0)
        total_labels= np.concatenate(total_labels, axis=0)
        train_acc = float(np.sum(total_preds==total_labels))/total_preds.shape[0]

        test_acc_ensemble, test_acc_last, test_acc_first, test_acc_second = evaluate(classifier, data_loader['test'])

        lr_current = max([param_group['lr'] for param_group in optimizer.param_groups])

        log_str = ['[%d/%d]  learing_rate: %.5f  loss: %.4f  train_acc: %.4f  test_acc_ensemble: %.4f   test_acc_last: %.4f   test_acc_first: %.4f  test_acc_second: %.4f' 
                   % (actual_steps, training_opt['num_steps'], lr_current, total_loss, train_acc, test_acc_ensemble, test_acc_last, test_acc_first, test_acc_second)]
        
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
            filename = "best_checkpoint.pth"
            filename = os.path.join(log_dir, filename)
            save_checkpoint(states, filename)
        
        filename = "last_checkpoint.pth"
        filename = os.path.join(log_dir, filename)
        save_checkpoint(states, filename)

        if actual_steps >= training_opt['num_steps']:
            break

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./config/classification/cifar100/vit_ensemble_no_pretrain.yaml', type=str)
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)

    args = parser.parse_args()

    mixer_types = ['mixup', 'cutmix']
    #vit_types = ['vit_base_patch32_384']

    #for mixer_type in mixer_types:    #testing with different mixup methods
        #for vit_type in vit_types:    #testing with vit types 
    with open(args.cfg) as f:
        config = yaml.safe_load(f)
    config['local_rank'] = args.local_rank
    #config['training_opt']['mixer_type'] = mixer_type
    #config['networks']['params']['model_name'] = vit_type

    train(config)
            
    print('ALL COMPLETED.')
