import numpy as np
import os, random
import torch

class CFG:
    seed          = 101
    debug         = False # set debug=False for Full Training
    exp_name      = 'Unet-For-UWMGI'
    comment       = 'Unet-resnet18-224x224-UWMGI'
    model_name    = 'Unet'
#     backbone      = 'efficientnet-b1' # 原 backbone
    backbone      = 'resnet18'
    train_bs      = 64
    valid_bs      = train_bs * 2
    img_size      = [224, 224]
    epochs        = 15
    lr            = 1e-3
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    T_max         = int (30000 / train_bs * epochs) + 50
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    n_accumulate  = max (1, 32 // train_bs)
    n_fold        = 5
    num_classes   = 3
    # device        = torch.device ("cuda:0" if torch.cuda.is_available () else "cpu")
    device = "cpu"

def set_seed (seed = 42):
    '''
        初始化各个随机种子为同一值, 
        保证每次执行程序的运行情况都相同
    '''
    np.random.seed (seed)
    random.seed (seed)
    torch.manual_seed (seed)
    torch.cuda.manual_seed (seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str (seed)
    print ('> SEEDING DONE')