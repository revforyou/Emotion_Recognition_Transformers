import os
import json
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
import transforms 
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, adjust_learning_rate, save_checkpoint
from train import train_epoch
from validation import val_epoch
import torchvision.transforms as tv_transforms  
import time
import random
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    opt = parse_opts()
    n_folds = 1
    test_accuracies = []
    
    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'  

    pretrained = opt.pretrain_path != 'None'    
    
    #opt.result_path = 'res_'+str(time.time())
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
        
    opt.arch = '{}'.format(opt.model)  
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])
            
    for fold in range(n_folds):

        print(opt)
        with open(os.path.join(opt.result_path, 'opts'+str(time.time())+str(fold)+'.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file)
            
        torch.manual_seed(opt.manual_seed)
        model, parameters = generate_model(opt)

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(opt.device)
        
        if not opt.no_train:
            
            video_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotate(),
                transforms.ToTensor(opt.video_norm_value)])
        
            training_data = get_training_set(opt, spatial_transform=video_transform) 
        
            train_loader = torch.utils.data.DataLoader(
                training_data,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=True)
        
            train_logger = Logger(
                os.path.join(opt.result_path, 'train'+str(fold)+'.log'),
                ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
            train_batch_logger = Logger(
                os.path.join(opt.result_path, 'train_batch'+str(fold)+'.log'),
                ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])
            

            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=opt.dampening,
                weight_decay=opt.weight_decay,
                nesterov=False)
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.lr_patience)
            
        if not opt.no_val:
            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])     
        
            validation_data = get_validation_set(opt, spatial_transform=video_transform)
            
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        
            val_logger = Logger(
                    os.path.join(opt.result_path, 'val'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])
            test_logger = Logger(
                    os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])

            
        best_prec1 = 0
        best_loss = 1e10
        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path, map_location=torch.device('cpu'))
            
            # 检查模型架构是否匹配
            assert opt.arch == checkpoint['arch'], f"Model architecture mismatch: {opt.arch} != {checkpoint['arch']}"
            
            # 获取模型的预训练权重
            pre_trained_dict = checkpoint['state_dict']
            
            # 移除 "module." 前缀（如果存在）
            pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
            
            # 获取当前模型的 state_dict
            model_dict = model.state_dict()
            
            # 过滤掉不匹配的权重
            filtered_dict = {key: value for key, value in pre_trained_dict.items() if key in model_dict}
            
            # 更新模型的 state_dict
            model_dict.update(filtered_dict)
            
            # 加载更新后的权重到模型
            model.load_state_dict(model_dict)
            
            # 更新训练相关信息
            best_prec1 = checkpoint.get('best_prec1', 0)  # 预防未定义 'best_prec1' 的情况
            opt.begin_epoch = checkpoint.get('epoch', 1)  # 如果未定义 'epoch'，则默认为 1


        for i in range(opt.begin_epoch, opt.n_epochs + 1):

            if not opt.no_train:
                adjust_learning_rate(optimizer, i, opt)
                train_epoch(i, train_loader, model, criterion, optimizer, opt,
                            train_logger, train_batch_logger)
                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1
                    }
                save_checkpoint(state, False, opt, fold)
            
            if not opt.no_val:
                
                validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, opt,
                                            val_logger)
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
               
                save_checkpoint(state, is_best, opt, fold)

               
        if opt.test:
            print("Starting test...")
            test_logger = Logger(
                    os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])

            video_transform = tv_transforms.Compose([
                tv_transforms.Resize((224, 224)),         # 调整图像大小到模型需要的尺寸
                tv_transforms.ToTensor(),                 # 将 PIL Image 转换为 Tensor 格式
                tv_transforms.Normalize(mean=[0.5], std=[0.5])  # 根据预训练模型的需求进行标准化
            ])
                
            test_data = get_test_set(opt, spatial_transform=video_transform) 

        
            # 在test阶段，直接加载resume_path指定的模型权重
            if opt.test and opt.resume_path:
                print(f"Loading model from resume_path: {opt.resume_path}")
                checkpoint = torch.load(opt.resume_path, map_location=torch.device(opt.device))
                
                pre_trained_dict = checkpoint['state_dict']
                pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
                
                model_dict = model.state_dict()
                filtered_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
                model_dict.update(filtered_dict)
                model.load_state_dict(model_dict)
            else:
                # 原有逻辑，用于非test阶段
                best_model_path = '%s/%s_best' % (opt.result_path, opt.store_name) + str(fold) + '.pth'
                if os.path.exists(best_model_path):
                    print(f"Loading best model from: {best_model_path}")
                    best_state = torch.load(best_model_path, map_location=torch.device(opt.device))
                    model.load_state_dict(best_state['state_dict'])
                else:
                    print(f"Warning: Best model file {best_model_path} not found. Using default pretrained model.")

        
            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
            
            test_loss, test_prec1 = val_epoch(10000, test_loader, model, criterion, opt,
                                            test_logger)
            
            with open(os.path.join(opt.result_path, 'test_set_bestval'+str(fold)+'.txt'), 'a') as f:
                    f.write('Prec1: ' + str(test_prec1) + '; Loss: ' + str(test_loss))
            test_accuracies.append(test_prec1) 

            print("Test completed.")
                
            
    with open(os.path.join(opt.result_path, 'test_set_bestval.txt'), 'a') as f:
        f.write('Prec1: ' + str(np.mean(np.array(test_accuracies))) +'+'+str(np.std(np.array(test_accuracies))) + '\n')
