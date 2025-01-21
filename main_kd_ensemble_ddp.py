import argparse
import yaml

import numpy as np
import functools
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
from datasets.utils import get_dataloader
from models.utils import *

import importlib
from tqdm import tqdm

import torch.distributed as dist

from models.KD import cifar_model_dict, imagenet_model_dict

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True

def get_loss2(logits, logits_mixed, features, features_mixed, y_a, y_b=None, logits_teacher=None,
                mixed_loss=True, weights=None, base_weight=1, 
                gamma=0.5, lam=1.0, ce_weight=1.0, kd_weight=1.0, tempture=4.0):
    
    #step 1: feature constrastive learning loss
    batch_size = logits.shape[0]
    features = features / features.norm(dim=1, keepdim=True)
    features_mixed = features_mixed / features_mixed.norm(dim=1, keepdim=True)
    #borrowed from CLIP
    features_logits = features @ features_mixed.t()
    features_logits = features_logits.unsqueeze(2)
    
    #if weights is None:
    weights = features_logits
    #else:
    #    weights = torch.cat((weights, features_logits), 2)
    modulating_factor = torch.mean(weights, dim=2).view(batch_size, -1)
    modulating_factor = torch.softmax(modulating_factor, dim=-1)
    #features_pt = torch.softmax(features_logits, dim=-1)
    features_ground_truth = torch.arange(batch_size, dtype=torch.long).view(-1, 1).to(logits.device)
    #step 2: supervised learning loss
    modulating_factor = modulating_factor.gather(1, features_ground_truth)
    
    
    if mixed_loss:
        #step 1: calculating CE loss
        log_student_ce = F.log_softmax(logits_mixed, dim=1)
        ce_loss = lam*log_student_ce.gather(1, y_a.view(-1, 1)) + (1-lam)*log_student_ce.gather(1, y_b.view(-1, 1))
        ce_loss = -1*ce_loss.view(-1, 1)

        #step 2: calculating KD_Loss
        log_student_kd = F.log_softmax(logits_mixed/tempture, dim=1)
        prob_teacher = F.softmax(logits_teacher/tempture, dim=1)
        kd_loss = -torch.sum(prob_teacher*log_student_kd, dim=-1)*(tempture**2)
        kd_loss = kd_loss.view(-1, 1)

        #step 3: final loss
        loss = (base_weight+modulating_factor)**gamma * (ce_weight*ce_loss + kd_weight*kd_loss)
        loss = loss.mean()

    else:
        log_student_ce = F.log_softmax(logits, dim=1)
        ce_loss = log_student_ce.gather(1, y_a.view(-1, 1))
        ce_loss = -1*ce_loss.view(-1, 1)

        #step 2: calculating KD_Loss
        log_student_kd = F.log_softmax(logits/tempture, dim=1)
        prob_teacher = F.softmax(logits_teacher/tempture, dim=1)
        kd_loss = -torch.sum(prob_teacher*log_student_kd, dim=-1)*(tempture**2)
        kd_loss = kd_loss.view(-1, 1)

        #step 3: final loss
        loss = (base_weight+modulating_factor)**gamma * (ce_weight*ce_loss + kd_weight*kd_loss)
        loss = loss.mean()

    #print('CE_Loss=', loss.item(), "   feature_loss=", features_loss.item())
    return loss, weights

def get_loss3(logits, logits_mixed, features, features_mixed, y_a, y_b=None, logits_teacher=None,
                mixed_loss=True, weights=None, base_weight=1, 
                gamma=0.5, lam=1.0, ce_weight=1.0, kd_weight=1.0, tempture=4.0):
    
    #step 1: feature constrastive learning loss
    batch_size = logits.shape[0]
    features = features / features.norm(dim=1, keepdim=True)
    features_mixed = features_mixed / features_mixed.norm(dim=1, keepdim=True)
    #borrowed from CLIP
    features_logits = features @ features_mixed.t()
    features_logits = features_logits.unsqueeze(2)
    
    #if weights is None:
    weights = features_logits
    #else:
    #    weights = torch.cat((weights, features_logits), 2)
    modulating_factor = torch.mean(weights, dim=2).view(batch_size, -1)
    modulating_factor = torch.softmax(modulating_factor, dim=-1)
    #features_pt = torch.softmax(features_logits, dim=-1)
    features_ground_truth = torch.arange(batch_size, dtype=torch.long).view(-1, 1).to(logits.device)
    #step 2: supervised learning loss
    modulating_factor = modulating_factor.gather(1, features_ground_truth)
    
    
    if mixed_loss:
        #step 1: calculating CE loss
        log_student_ce = F.log_softmax(logits_mixed, dim=1)
        ce_loss = lam*log_student_ce.gather(1, y_a.view(-1, 1)) + (1-lam)*log_student_ce.gather(1, y_b.view(-1, 1))
        ce_loss = -1*ce_loss.view(-1, 1)

        #step 2: calculating KD_Loss
        log_student_kd = F.log_softmax(logits_mixed/tempture, dim=1)
        prob_teacher = F.softmax(logits_teacher/tempture, dim=1)
        #kd_loss = -torch.sum(prob_teacher*log_student_kd, dim=-1)*(tempture**2)
        #kd_loss = kd_loss.view(-1, 1)
        kd_loss = F.kl_div(log_student_kd, prob_teacher, reduction="none").sum(1)
        kd_loss = kd_loss*(tempture**2)
        kd_loss = kd_loss.view(-1, 1)

        #step 3: final loss
        loss = (base_weight+modulating_factor)**gamma * (ce_weight*ce_loss + kd_weight*kd_loss)
        loss = loss.mean()

    else:
        log_student_ce = F.log_softmax(logits, dim=1)
        ce_loss = log_student_ce.gather(1, y_a.view(-1, 1))
        ce_loss = -1*ce_loss.view(-1, 1)

        #step 2: calculating KD_Loss
        log_student_kd = F.log_softmax(logits/tempture, dim=1)
        prob_teacher = F.softmax(logits_teacher/tempture, dim=1)
        kd_loss = -torch.sum(prob_teacher*log_student_kd, dim=-1)*(tempture**2)
        kd_loss = kd_loss.view(-1, 1)

        #step 3: final loss
        loss = (base_weight+modulating_factor)**gamma * (ce_weight*ce_loss + kd_weight*kd_loss)
        loss = loss.mean()

    #print('CE_Loss=', loss.item(), "   feature_loss=", features_loss.item())
    return loss, weights


def evaluate(classifier, test_loader, teacher_model, device='cuda'):
    torch.cuda.empty_cache()
    classifier.eval()
    teacher_model.eval()
    total_preds_teacher, total_preds, total_labels, total_preds_last, total_preds_first, total_preds_second = [], [], [], [], [], []

    for data in tqdm(test_loader):
        real_x = data[0].to(device)
        real_y = data[1].to(device)

        with torch.no_grad():
            logits, _ = classifier(real_x)
            logits_teacher, _ = teacher_model(real_x)

            logits_ensemble = 0
            
            for i in range(len(logits)):
                logits_ensemble += torch.softmax(logits[i], dim=1)

            #_, pred = torch.max(logits_ensemble, dim=1)
            total_preds.append(logits_ensemble)
            total_labels.append(real_y)

            #_, pred_last = torch.max(logits[-1], dim=1)
            total_preds_last.append(logits[-1])

            #_, pred_first = torch.max(logits[0], dim=1)
            total_preds_first.append(logits[0])

            #_, pred_second = torch.max(logits[1], dim=1)
            total_preds_second.append(logits[1])

            #_, pred_teacher = torch.max(logits_teacher, dim=1)
            total_preds_teacher.append(logits_teacher)

    total_preds = torch.cat(total_preds, axis=0)
    total_labels= torch.cat(total_labels, axis=0)
    total_preds_last = torch.cat(total_preds_last, axis=0)
    total_preds_first = torch.cat(total_preds_first, axis=0)
    total_preds_second = torch.cat(total_preds_second, axis=0)
    total_preds_teacher = torch.cat(total_preds_teacher, axis=0)

    test_acc1, test_acc5 = accuracy(total_preds, total_labels, topk=(1, 5))
    test_acc_last1, test_acc_last5 = accuracy(total_preds_last, total_labels, topk=(1, 5))
    test_acc_first1, test_acc_first5 = accuracy(total_preds_first, total_labels, topk=(1, 5))
    test_acc_second1, test_acc_second5 = accuracy(total_preds_second, total_labels, topk=(1, 5))
    test_acc_teacher1, test_acc_teacher5 = accuracy(total_preds_teacher, total_labels, topk=(1, 5))
    
    return test_acc1, test_acc5, test_acc_last1, test_acc_last5, test_acc_first1, test_acc_first5, test_acc_second1, test_acc_second5, test_acc_teacher1, test_acc_teacher5
    

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
        data_loader = get_dataloader(distributed=False, options=dataset_cofig)
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

    log_dir = str(training_opt['log_dir'])
    log_dir = os.path.join(log_dir, str(training_opt['mixer_type']))
    
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'log.txt')
    
   
    if training_opt['mixer_type']=='mixup':
        data_mixer = mixup_data
    elif training_opt['mixer_type']=='cutmix':
        data_mixer = cutmix
    #device = torch.device("cuda")   
    
    #step 2: prepare teacher & student model
    if dataset_cofig['dataset']=='cifar100':
        net, pretrain_model_path = cifar_model_dict[networks_args['teacher']]
        teacher_model = net(num_classes=networks_args['student']['params']['num_classes'])
        model_dict = torch.load(pretrain_model_path, map_location='cpu')
        teacher_model.load_state_dict(model_dict["model"])

    elif dataset_cofig['dataset'].lower()=='imagenet':
        teacher_model = imagenet_model_dict[networks_args['teacher']]
    
    classifier_args = config['networks']['student']
    classifier_def_file = classifier_args['def_file']
    classifier = source_import(classifier_def_file).create_model(**classifier_args['params'])
    
    if rank==-1:
        teacher_model = nn.DataParallel(teacher_model).to(device)
        classifier = nn.DataParallel(classifier).to(device)
    else:
        teacher_model = nn.parallel.DistributedDataParallel(teacher_model.to(device), broadcast_buffers=False, device_ids=[rank], output_device=rank)
        classifier = nn.parallel.DistributedDataParallel(classifier.to(device), broadcast_buffers=False, device_ids=[rank], output_device=rank)
        
    
    teacher_model.eval()
    
    #step 4: Optimizer & scheduler
    optimizer = optim.SGD(classifier.parameters(), lr=optimizer_args['lr'], 
                          momentum=optimizer_args['momentum'], 
                          weight_decay=optimizer_args['weight_decay'])
    
    if config['coslr']:
        print("===> Using coslr eta_min={}".format(config['endlr']))
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=training_opt['warmup_epoch'], t_total=training_opt['num_epochs'])

        
    else:
        print("===> Using multistepLR eta_min={}")
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=training_opt['milestones'])
        

    acc_record, total_steps = 0, 0
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
            real_x = data[0].to(device)
            real_y = data[1].to(device)
            total_steps = total_steps + 1
            mixed_x, y_a, y_b, lam, _ = data_mixer(real_x, real_y, alpha=training_opt['alpha'])

            logits_mixed, features_mixed = classifier(mixed_x)
            logits, features = classifier(real_x)
            
            with torch.no_grad():
                if training_opt['mixed_loss']:
                    logits_teacher, _ = teacher_model(mixed_x)
                else:
                    logits_teacher, _ = teacher_model(real_x)
            
            #soft_targets = torch.softmax(teacher_logits, dim=-1)
            weights = None
            loss = 0
            logits_ensemble = 0


            for i in range(0, len(logits)):
                loss_i, weights = get_loss3(logits=logits[i], logits_mixed=logits_mixed[i], 
                                        features=features[i], features_mixed=features_mixed[i], 
                                        y_a=y_a, y_b=y_b, 
                                        logits_teacher=logits_teacher,
                                        mixed_loss=training_opt['mixed_loss'],
                                        weights=weights, 
                                        base_weight=training_opt['base_weight'], 
                                        gamma=training_opt['gamma'], 
                                        lam=lam, 
                                        ce_weight=training_opt['ce_weight'], 
                                        kd_weight=training_opt['kd_weight'], 
                                        tempture=training_opt['tempture'])

                logits_ensemble += torch.softmax(logits[i], dim=1)
                loss = loss + loss_i
            
            loss.backward()
            total_loss += loss.item()

            if total_steps % training_opt['num_accmutations']==0 or total_steps % len(data_loader['train'])==0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            #_, pred = torch.max(logits_ensemble, dim=1)
            total_preds.append(logits_ensemble.detach().clone())
            total_labels.append(real_y)
           

        total_loss /= len(data_loader['train'])
        total_preds = torch.cat(total_preds, axis=0)
        total_labels= torch.cat(total_labels, axis=0)
        train_acc1, train_acc5 = accuracy(total_preds, total_labels, topk=(1, 5))

        test_acc1, test_acc5, \
            test_acc_last1, test_acc_last5, \
            test_acc_first1, test_acc_first5, \
            test_acc_second1, test_acc_second5, \
            test_acc_teacher1, test_acc_teacher5 = evaluate(classifier, data_loader['test'], teacher_model)

        lr_current = max([param_group['lr'] for param_group in optimizer.param_groups])

        log_str = ['[%d/%d]  learing_rate: %.5f  loss: %.4f  test_acc_teacher1:%.4f  test_acc_teacher5:%.4f  test_acc_ensemble1: %.4f  test_acc_ensemble5: %.4f  test_acc_last1: %.4f  test_acc_last5: %.4f' 
                   % (epoch, training_opt['num_epochs'], lr_current, total_loss, \
                        test_acc_teacher1, test_acc_teacher5, \
                        test_acc1, test_acc5, \
                        test_acc_last1, test_acc_last5)]
        
        
        #print(log_str)
        print_write(log_str, log_file)
    
        states = {
            'epoch':epoch,
            'acc': train_acc1,
            'test_acc': test_acc1,
            'classifier_dict':classifier.state_dict(),
            'optimizer':optimizer.state_dict()
        }

        if train_acc1>acc_record:
            acc_record = train_acc1
            #filename = "best_checkpoint.pth"
            filename = "best_checkpoint_%d.pth" % (rank)
            filename = os.path.join(log_dir, filename)
            save_checkpoint(states, filename)
        
        filename = "last_checkpoint.pth"
        filename = os.path.join(log_dir, filename)
        save_checkpoint(states, filename)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./config/kd/cifar100/resnet50_mobilenetv2.yaml', type=str)
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
    args = parser.parse_args()
    
    mixer_types = ['mixup']
    
    for mixer_type in mixer_types:    #testing with different mixup methods
        with open(args.cfg) as f:
            config = yaml.safe_load(f)
        
        config['local_rank'] = args.local_rank
        config['training_opt']['mixer_type'] = mixer_type
        
    train(config)
            
    print('ALL COMPLETED.')