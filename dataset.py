from utility import data_transforms, load_img, load_msk
from config import CFG
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
import torch

class BuildDataset (torch.utils.data.Dataset):
    '''
        数据集总体可以分为两类, 有标签和无标签
        对于训练集, 需要进行一系列数据增强的操作, 包括图片的旋转、水平翻转
        对于测试集和验证集, 则不需要数据增强
        所以需要设计两种 transforms
        训练集、测试集、验证集均需要固定图片尺寸 (224, 224)
    '''
    def __init__(self, df, label = True, transforms = None) :
        self.df         = df
        self.label      = label
        self.img_paths  = df['image_path'].tolist()
        self.msk_paths  = df['mask_path'].tolist()
        self.transforms = transforms
        
    def __len__(self) :
        return len (self.df)
    
    def __getitem__(self, index) :
        img_path  = self.img_paths[index]
        # img_path = img_path.replace ("/kaggle/input", ".")
        img = []
        img = load_img (img_path)
        
        if self.label :
            msk_path = self.msk_paths[index]
            # msk_path = msk_path.replace ("/kaggle/input", ".")
            msk = load_msk (msk_path)
            if self.transforms:
                data = self.transforms (image = img, mask = msk)
                img  = data['image']
                msk  = data['mask']
            img = np.transpose (img, (2, 0, 1))
            msk = np.transpose (msk, (2, 0, 1))
            return torch.tensor (img), torch.tensor (msk)
        else :
            if self.transforms :
                data = self.transforms (image = img)
                img  = data['image']
            img = np.transpose (img, (2, 0, 1))
            return torch.tensor (img)

def DataLoader (fold) :
    '''
        K 折交叉验证, 指定验证集为哪一折, 划分训练集和验证集
    '''
    # drop 参数表示是否删除原来的索引列
    train_df = df.query("fold!=@fold & empty==0").reset_index (drop = True)
    valid_df = df.query("fold==@fold & empty==0").reset_index (drop = True)
    train_dataset = BuildDataset (train_df, transforms = data_transforms['train'])
    valid_dataset = BuildDataset (valid_df, transforms = data_transforms['valid'])

    # pin_memory 参数表示是否将加载的数据常驻内存
    # drop_last 参数表示是否丢弃最后一个批次 (可能不满 batch_size 个样本)
    train_loader = torch.utils.data.DataLoader (train_dataset, batch_size = CFG.train_bs, num_workers = 4, 
                               shuffle = True, pin_memory = True, drop_last = False)
    valid_loader = torch.utils.data.DataLoader (valid_dataset, batch_size = CFG.valid_bs, num_workers = 4, 
                               shuffle = False, pin_memory = True)
    
    return train_loader, valid_loader

df = pd.read_csv('./uwmgi-mask-dataset/train.csv')
# 填充表格中空着的 segmentation 项
df['segmentation'] = df.segmentation.fillna('') 
df['image_path'] = df.image_path.str.replace ('/kaggle/input', '.')
df['mask_path'] = df.mask_path.str.replace ('/kaggle/input', '.')
# 把掩码对应的图片地址修改成对应的 npy 地址
df['mask_path'] = df.mask_path.str.replace('/png/','/np').str.replace('.png','.npy') 
# 计算每一个样本的掩码的长度, 用来判断该样本是否有掩码
df['rle_len'] = df.segmentation.map (len) 
# 计算每个样本对应的掩码长度之和, 只要有一个及以上类别存在掩码, 即判断该图像有掩码, 同时把样本对应的三项归为一项了
df2 = df.groupby (['id'])['rle_len'].agg (sum).to_frame ().reset_index () 
# 添加 empty 列, 说明该样本的状态 (即是否有掩码)
df2['empty'] = (df2.rle_len == 0) 
df2 = df2.drop (columns = ['rle_len'])
# 删除无用信息
df = df.drop (columns = ['class', 'segmentation', 'day', 'slice', 'height', 'width', 'rle_len']) 
# 三项归为一项
df = df.drop_duplicates (subset = ['id'], keep = 'first') 
# 获取 empty 状态
df = df.merge (df2, on = ['id']) 

# 删除脏数据
Case138_Day0 = [i for i in range (76, 145)]
Case85_Day23 = [119,120,121,122,123,124]
Case90_Day29 = [115,116,117,118,119]
Case133_Day25 = [111,112,113]
Case7 = []
Case43 = []
Case81 = []
Case85 = []
Case90 = []
Case133 = []
Case138 = []
for i,row in tqdm (df.iterrows (), total = len (df)) :
    if row.id.rsplit ("_",2)[0] == 'case7_day0':
        Case7.append (i)
    elif row.id.rsplit ("_",2)[0] == 'case43_day18' or row.id.rsplit ("_",2)[0] == 'case43_day26' :
        Case43.append (i)
    elif row.id.rsplit ("_",2)[0] == 'case81_day30' :
        Case81.append (i)
    elif row.id.rsplit ("_",2)[0] == 'case138_day0' :
        if int (row.id.rsplit ("_",1)[-1]) in Case138_Day0 :
            Case138.append (i)
df.drop (index = Case7 + Case43 + Case81 + Case138 ,inplace = True)
df = df.reset_index (drop = True)

skf = StratifiedGroupKFold (n_splits = CFG.n_fold, shuffle = True, random_state = CFG.seed)
for fold, (train_idx, val_idx) in enumerate (skf.split (df, df['empty'], groups = df["case"])):
    df.loc[val_idx, 'fold'] = fold