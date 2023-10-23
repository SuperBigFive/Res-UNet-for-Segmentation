from config import CFG, set_seed
from dataset import DataLoader
from model import build_model

import numpy as np

import os
# tqdm 库的功能是提供一个快速，可扩展的 Python 进度条，可以在 Python 长循环中添加一个进度提示信息
from tqdm import tqdm
tqdm.pandas ()
import time
import copy
from collections import defaultdict

# 强制回收垃圾, 释放空间
import gc

# Pytorch 常用库
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
# from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

# 终端输出是可以输出不同颜色的文字
from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings ("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import segmentation_models_pytorch as smp

BCELoss     = smp.losses.SoftBCEWithLogitsLoss ()
TverskyLoss = smp.losses.TverskyLoss (mode = 'multilabel', log_loss = False)

def dice_coef (y_true, y_pred, thr = 0.5, dim = (2,3), epsilon = 0.001) :
    '''
        图像 A, B 的 dice 系数等于 A 和 B 的掩码区域的交集的面积大小乘以 2
        再除以 A 的掩码区域面积与 B 的掩码区域面积的和
        可以用来衡量两个图像的相似程度
    '''
    y_true = y_true.to (torch.float32)
    y_pred = (y_pred > thr).to (torch.float32)
    inter = (y_true * y_pred).sum (dim = dim)
    den = y_true.sum (dim = dim) + y_pred.sum (dim = dim)
    dice = ((2 * inter + epsilon) / (den+epsilon)).mean (dim = (1,0))
    return dice

def iou_coef (y_true, y_pred, thr = 0.5, dim = (2, 3), epsilon = 0.001) :
    '''
        iou 系数与 dice 系数类似, 计算公式中分子均为两图像的交集, 
        dice 系数的计算公式的分母为两图像面积和
        iou 系数的计算公式的分母为两图像并集
    '''
    y_true = y_true.to (torch.float32)
    y_pred = (y_pred > thr).to (torch.float32)
    inter = (y_true * y_pred).sum (dim = dim)
    union = (y_true + y_pred - y_true * y_pred).sum (dim = dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean (dim = (1,0))
    return iou

def criterion (y_pred, y_true) :
    return 0.5 * BCELoss (y_pred, y_true) + 0.5 * TverskyLoss (y_pred, y_true)

def train_one_epoch (model, optimizer, scheduler, dataloader, device, epoch) :
    # model.train () 与 model.eval () 主要影响网络中 BatchNorm 层和 Dropout 层
    # 前者启用, 后者不启用; model.eval () 模式下不会进行反向传播, 但是梯度的计算照常进行
    # model.eval () 配合 torch.no_grad () 使用, 加速计算过程、节省显存空间
    model.train ()
    # 创建一个 GradScaler 对象, 可以在迭代过程中动态估计损失放大的倍数
    scaler = amp.GradScaler ()
    
    dataset_size = 0
    running_loss = 0.0
    
    # 将一个可迭代对象作为参数传入，然后返回一个包装后的可迭代对象，
    # 可以像平常一样对其进行迭代，每次请求一个值时，都会打印一个进度条。
    pbar = tqdm (enumerate (dataloader), total = len (dataloader), desc = 'Train ')
    for step, (images, masks) in pbar:         
        images = images.to (device, dtype = torch.float)
        masks  = masks.to (device, dtype = torch.float)
        
        batch_size = images.size (0)
        
        # 前向传播过程中自动混合精度训练
        with amp.autocast (enabled = True):
            y_pred = model (images)
            loss   = criterion (y_pred, masks)
            # n_accumulate 参数的含义是每若干个批次后进行一次梯度更新
            loss   = loss / CFG.n_accumulate
        # 放大损失、反向传播
        scaler.scale (loss).backward ()
    
        if (step + 1) % CFG.n_accumulate == 0 :
            # 根据原放大倍数，梯度更新时缩小相应的倍数
            scaler.step (optimizer)
            # 更新损失放大的倍数
            scaler.update ()

            optimizer.zero_grad ()

            if scheduler is not None :
                # 更新学习率
                scheduler.step ()
                
        running_loss += (loss.item () * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved () / 1E9 if torch.cuda.is_available () else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix (train_loss = f'{epoch_loss : 0.4f}',
                        lr = f'{current_lr : 0.5f}',
                        gpu_mem = f'{mem : 0.2f} GB')
    torch.cuda.empty_cache ()
    gc.collect ()
    
    return epoch_loss

@torch.no_grad ()
def valid_one_epoch (model, dataloader, device, epoch):
    model.eval ()
    
    dataset_size = 0
    running_loss = 0.0
    
    val_scores = []
    
    pbar = tqdm (enumerate (dataloader), total = len (dataloader), desc = 'Valid ')
    for step, (images, masks) in pbar :
        images  = images.to (device, dtype = torch.float)
        masks   = masks.to (device, dtype = torch.float)
        
        batch_size = images.size (0)
        
        y_pred  = model (images)
        loss    = criterion (y_pred, masks)
        
        running_loss += (loss.item () * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        y_pred = nn.Sigmoid () (y_pred)
        val_dice = dice_coef (masks, y_pred).cpu ().detach ().numpy ()
        val_jaccard = iou_coef (masks, y_pred).cpu ().detach ().numpy ()
        val_scores.append ([val_dice, val_jaccard])
        
        mem = torch.cuda.memory_reserved () / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix (valid_loss = f'{epoch_loss : 0.4f}',
                        lr = f'{current_lr : 0.5f}',
                        gpu_memory = f'{mem : 0.2f} GB')
    val_scores  = np.mean (val_scores, axis = 0)
    torch.cuda.empty_cache ()
    gc.collect()
    
    return epoch_loss, val_scores

def run_training (model, optimizer, scheduler, device, num_epochs) :
    '''
        这里主要做一些记录训练日志、保存最优模型等工作
    '''
    
    if torch.cuda.is_available ():
        print("cuda: {}\n".format(torch.cuda.get_device_name ()))
    
    start = time.time ()
    best_model_wts = copy.deepcopy (model.state_dict())
    best_dice      = -np.inf
    best_epoch     = -1
    history = defaultdict (list)
    
    for epoch in range (1, num_epochs + 1): 
        gc.collect ()
        print (f'Epoch {epoch} / {num_epochs}', end = '')
        train_loss = train_one_epoch (model, optimizer, scheduler, 
                                           dataloader = train_loader, 
                                           device = CFG.device, epoch = epoch)
        
        val_loss, val_scores = valid_one_epoch (model, valid_loader, 
                                                 device = CFG.device, 
                                                 epoch = epoch)
        val_dice, val_jaccard = val_scores
    
        history['Train Loss'].append (train_loss)
        history['Valid Loss'].append (val_loss)
        history['Valid Dice'].append (val_dice)
        history['Valid Jaccard'].append (val_jaccard)
      
        
        print(f'Valid Dice: {val_dice : 0.4f} | Valid Jaccard: {val_jaccard : 0.4f}')
        
        if val_dice >= best_dice:
            print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice    = val_dice
            best_jaccard = val_jaccard
            best_epoch   = epoch
            best_model_wts = copy.deepcopy (model.state_dict ())
            PATH = f"best_epoch-{fold:02d}.bin"
            torch.save (model.state_dict (), PATH)
            print (f"Model Saved{sr_}")
            
        last_model_wts = copy.deepcopy (model.state_dict ())
        PATH = f"last_epoch-{fold : 02d}.bin"
        torch.save (model.state_dict (), PATH)
            
        print (); print ()
    
    end = time.time()
    time_elapsed = end - start
    print ('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format (
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print ("Best Score: {:.4f}".format (best_jaccard))
    
    model.load_state_dict (best_model_wts)
    
    return model, history

def fetch_scheduler (optimizer):
    if CFG.scheduler == 'CosineAnnealingLR' :
        scheduler = lr_scheduler.CosineAnnealingLR (optimizer, T_max = CFG.T_max, 
                                                   eta_min = CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts' :
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts (optimizer, T_0 = CFG.T_0, 
                                                             eta_min = CFG.min_lr)
    elif CFG.scheduler == 'ReduceLROnPlateau' :
        scheduler = lr_scheduler.ReduceLROnPlateau (optimizer,
                                                   mode = 'min',
                                                   factor = 0.1,
                                                   patience = 7,
                                                   threshold = 0.0001,
                                                   min_lr = CFG.min_lr,)
    elif CFG.scheduer == 'ExponentialLR' :
        scheduler = lr_scheduler.ExponentialLR (optimizer, gamma = 0.85)
    elif CFG.scheduler == None:
        return None
        
    return scheduler

if __name__ == "__main__" :
    set_seed (CFG.seed)
    fold = 0
    train_loader, valid_loader = DataLoader (fold = fold)
    model     = build_model ()
    optimizer = optim.Adam (model.parameters(), lr = CFG.lr, weight_decay = CFG.wd)
    scheduler = fetch_scheduler (optimizer)
    model, history = run_training (model, optimizer, scheduler,
                                  device = CFG.device,
                                  num_epochs = 1)