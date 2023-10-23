from config import CFG
from dataset import BuildDataset, DataLoader, df
from utility import show_img, png2tensor, plot_batch, data_transforms
from model import load_model
import torch
from torch import nn
import torch.functional as F
import matplotlib.pyplot as plt
import random, shutil

def Predict (model, imgs, msks, size, plot_img = True) :
    '''
        输入模型, tensor 类型的图像, 标签, 即个数,
        输出三行(或两行)图像, 
        第一行只有图像, 
        第二行为图像和标签的重叠,
        第三行为图像和预测标签的重叠
    '''
    imgs = imgs.to (CFG.device, dtype = torch.float)
    msks = msks.cpu ().detach () if not msks == None else None
    with torch.no_grad () :
        preds = model (imgs)
        # 事实上这一步仅仅是用来把所有大于 0 的元素都置 1 了
        # 因为类别仅有三种, 刚好可以分别用三个通道单独表示, 标签也是这么做的
        # preds = (nn.Sigmoid ()(preds) > 0.5).double ().cpu ().detach ()
        preds = (nn.Sigmoid ()(preds)).double ().cpu ().detach ()
        for idx, pred in enumerate (preds) :
            pred = pred.permute (1, 2, 0)
            label = torch.argmax (pred, dim = 2)
            pred = (pred > 0.5)
            npred = F.one_hot (label, CFG.num_classes)
            pred = pred * npred
            pred = pred.permute (2, 0, 1)
            preds[idx] = pred
    imgs = imgs.cpu ().detach ()
    if not plot_img : return preds
    plot_batch (imgs, torch.zeros_like (imgs).cpu (), size = size)
    if not msks == None : plot_batch (imgs, msks, size = size)
    plot_batch (imgs, preds, size = size)

def GetPartImageAndMask (img, msk, lef_top = (32, 96), rig_bot = (128, 192)) :
    '''
        截取一对样本、标签的一个矩形区域
        输入 3D tensor, shape = (通道数, 长, 宽)
        lef_tor, tuple, 左上角坐标
        rig_bot, typle, 右下角坐标
    '''
    x0, y0 = lef_top
    x1, y1 = rig_bot
    img = img[: , x0 : x1, y0 : y1]
    msk = msk[: , x0 : x1, y0 : y1]
    lef_pad = rig_pad = (CFG.img_size[1] - (y1 - y0)) // 2
    top_pad = bot_pad = (CFG.img_size[0] - (x1 - x0)) // 2
    pad = (lef_pad, rig_pad, top_pad, bot_pad)
    img = F.pad (img, pad, mode = "constant", value = 0)
    msk = F.pad (msk, pad, mode = "constant", value = 0)
    return img, msk

def RandomSave (fold = 0, save_num = 5) :
    '''
        从数据集里随机找 save_num 个图像和标签, 保存在 output 里
    '''
    df_notEmpty = df.query("fold==@fold & empty==0").reset_index (drop = True)
    dataset = BuildDataset (df_notEmpty, transforms = data_transforms['valid'])
    for i in range (save_num) :
        img_PATH_new = f"slice_{i}.png"
        msk_PATH_new = f"slice_{i}.npy"
        id = random.randint (0, dataset.__len__ ())
        img_PATH_old = dataset.img_paths[id]
        msk_PATH_old = dataset.msk_paths[id]
#         print (img_PATH_new, img_PATH_old)
        shutil.copy (img_PATH_old, img_PATH_new)
        shutil.copy (msk_PATH_old, msk_PATH_new) 

def PredictWithLabel (fold, model, df) :
    '''
        使用有标签的图像进行测试
    '''
    # 选择第一折的有标签的数据作为测试集, 不进行数据增强
    test_dataset = BuildDataset (df.query ("fold==@fold & empty==0").sample (frac = 1.0), label = True, 
                            transforms = data_transforms['valid'])
    test_loader  = DataLoader (test_dataset, batch_size = 5, 
                            num_workers = 4, shuffle = False, pin_memory = True)
    # 获取一个批次大小 (5个) 的测试集
    imgs, msks = next (iter (test_loader))
    Predict (model, imgs, msks, 5)

def PredictWithNoLabel (model, df) :
    '''
        使用无标签的图像进行测试
    '''
    test_dataset = BuildDataset (df.query ("empty!=0").sample (frac = 1.0), label = False, 
                            transforms = data_transforms['valid'])
    test_loader  = DataLoader (test_dataset, batch_size = 5, 
                            num_workers = 4, shuffle = False, pin_memory = True)
    imgs = next (iter (test_loader)).to (CFG.device, dtype = torch.float)
    Predict (model, imgs, None, 5)

def PredictWithPart (model, size = 5) :
    '''
        随机截取若干张图像的局部图像, 并进行预测
    '''
    RandomSave (save_num = size)
    imgs, msks = [], []
    for i in range (size) :
        img_PATH = f"/kaggle/working/slice_{i}.png"
        msk_PATH = f"/kaggle/working/slice_{i}.npy"
        img, msk = png2tensor (img_PATH, msk_PATH)
        img, msk = GetPartImageAndMask (img, msk)
        imgs.append (img); msks.append (msk)
    imgs = torch.stack (imgs, dim = 0)
    msks = torch.stack (msks, dim = 0)
    Predict (model, imgs, msks, size)

def PredictionCompare (img, msk, models, save_path = None, pred_medsam = None) :
    '''
        输入图像和标签均为 tensor 类型
        对比朴素 unet、resnet50_unet、medsam 三种模型的语义分割效果
    '''
    figure_cnt = len (models) + 2
    plt.rcParams['figure.figsize'] = (5 * figure_cnt, 6)
#     titles = ["image", "mask", "unet", " unet-resnet50", "medsam"]
#     imgs = [img, msk, pred_unet, pred_unet_resnet50, pred_medsam]
    preds, titles = [], ['image', 'mask']
    img = img.unsqueeze (0)
    msk = msk.unsqueeze (0)
    for title, model in models.items () :
        preds.append (Predict (model, img, msk, 1, plot_img = False).squeeze ())
        titles.append (title)
    img = img.squeeze ()
    msk = msk.squeeze ()
    imgs = [img, msk] + preds
    if not pred_medsam == None :
        imgs.append (pred_medsam)
        titles.append ("medsam")
#     for idx, image in e
#         imgs[idx] = imgs[idx,].permute ((1, 2, 0)).numpy () * 255.
#     imgs[0] = imgs[0].astype ('uint8')
    
    for i, image in enumerate (imgs) :
        image = image.permute ((1, 2, 0)).numpy () * 255.0
        if i == 0 : image = image.astype ('uint8')
        imgs[i] = image
        plt.subplot (1, figure_cnt, i + 1)
        plt.title (titles[i], fontsize = "32")
        if i == 0 : show_img (img = imgs[0], mask = None)
        else : show_img (img = imgs[0], mask = image)
    plt.tight_layout ()
    if not save_path == None : plt.savefig (save_path); print (save_path)
    plt.show ()

model_unet_densenet161 = load_model ("densenet161", "models-for-unet/model_unet_densenet161.bin")
model_unet_efficientnet_b4 = load_model ("efficientnet-b4", "models-for-unet/model_unet_efficientnet_b4.bin")
model_unet_efficientnet_b5 = load_model ("efficientnet-b5", "models-for-unet/model_unet_efficientnet_b5.bin")
model_unet_mit_b2 = load_model ("mit_b2", "models-for-unet/model_unet_mit_b2.bin")
model_unet_mobilenet_v2 = load_model ("mobilenet_v2", "models-for-unet/model_unet_mobilenet_v2.bin")
model_unet_resnet101 = load_model ("resnet101", "models-for-unet/model_unet_resnet101.bin")
model_unet_resnet50 = load_model ("resnet50", "models-for-unet/model_unet_resnet50_2.bin")
model_unet_se_resnet50 = load_model ("se_resnet50", "models-for-unet/model_unet_se_resnet50.bin")
models = {"densenet161"     : model_unet_densenet161,
          "efficientnet-b4" : model_unet_efficientnet_b4,
          "efficientnet-b5" : model_unet_efficientnet_b5, 
          "mit-b2"          : model_unet_mit_b2,
          "mobilenet-v2"    : model_unet_mobilenet_v2,
          "resnet101"       : model_unet_resnet101,
          "resnet50"        : model_unet_resnet50,
          "se-resnet50"     : model_unet_se_resnet50}

# 取所有模型的集合的一个子集, 对同一张图片进行预测结果的可视化对比, 对比十次
# %mkdir -p "figures"
# model_cnt = len (models)
# key_value = list (models.items ())
# RandomSave (fold = random.randint (0, 4), save_num = 10)
# for i in range (1, 1 << model_cnt) :
#     dir_PATH = "figures/" + str (bindigits (i, model_cnt))
#     sub_models = dict ()
#     base_PATH = str ("/kaggle/working/") + str (dir_PATH) + str ("/")
#     cnt = 0
#     for j in range (model_cnt) :
#         if (i >> j) & 1 : 
#             sub_models[key_value[j][0]] = key_value[j][1]
#             if cnt > 0 : base_PATH += "|"
#             cnt += 1
#             base_PATH += str (key_value[j][0])
#     if cnt < 4 : continue
#     isExists = os.path.exists (dir_PATH)
#     if not isExists:
#         os.makedirs (dir_PATH)
#         print("%s 目录创建成功" % dir_PATH)
#     else:
#         print("目录已经存在")
#     for k in range (10) :
#         img_PATH = f"/kaggle/working/slice_{k}.png"
#         msk_PATH = f"/kaggle/working/slice_{k}.npy"
#         img, msk = png2tensor (img_PATH, msk_PATH)
#         img, msk = GetPartImageAndMask (img, msk)
#         save_PATH = base_PATH + f"|{k:02d}.png"
#         PredictionCompare (img, msk, sub_models, save_PATH)
#     gc.collect ()