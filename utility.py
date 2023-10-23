from config import CFG
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import albumentations as A
import zipfile, os

def load_img (path):
    '''
        根据文件路径读取图像, 并复制成三份, 成三通道图像
    '''
    img = cv2.imread (path, cv2.IMREAD_UNCHANGED)
    img = np.tile (img[...,None], [1, 1, 3]) # gray to rgb
    img = img.astype ('float32') # original is uint16
    mx = np.max (img)
    if mx:
        img /= mx # scale image to [0, 1]
    return img

def load_msk (path):
    '''
        根据文件路径读取掩码图像 (已经处理成 numpy 格式了)
    '''
    msk = np.load (path)
    msk = msk.astype('float32')
    msk /= 255.0
    return msk
    
def png2tensor (img_path, msk_path = None) :
    '''
        给定图片路径、掩码路径, 转为 tensor 类型
    '''
    img = load_img (img_path)
    if msk_path == None :
        data = data_transforms['valid'] (image = img)
        img = data['image']
        img = torch.tensor (np.transpose (img, (2, 0, 1)))
        return img
    else :
        msk = load_msk (msk_path)
        data = data_transforms['valid'] (image = img, mask = msk)
        img = data['image']
        msk = data['mask']
        img = torch.tensor (np.transpose (img, (2, 0, 1)))
        msk = torch.tensor (np.transpose (msk, (2, 0, 1)))
        return img, msk

def show_img (img, mask = None):
    ''' 
        展示图像, 如果输入了掩码图像则一起展示
        输入的图像要提前处理成 numpy 格式
    '''
#     用于医学影像增强
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     img = clahe.apply(img)
#     plt.figure(figsize=(10,10))
    plt.imshow (img, cmap = 'bone')
    
    if mask is not None:
        plt.imshow (mask, alpha = 0.5) # alpha 参数, 透明度
        handles = [Rectangle ((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend (handles, labels) # 标签与掩码颜色相对应
    plt.axis('off')

def plot_batch (imgs, msks, size = 3) :
    '''
        输入的 tensor 类型的数据, 要先把通道维度换到最后一维, 然后转成 numpy 类型
    '''
    plt.figure (figsize = (5 * size, 5))
    for idx in range (size) :
        plt.subplot (1, size, idx + 1)
        img = imgs[idx,].permute ((1, 2, 0)).numpy () * 255.0
        img = img.astype ('uint8')
        msk = msks[idx,].permute ((1, 2, 0)).numpy () * 255.0
        show_img (img, msk)
    plt.tight_layout ()
    plt.show ()

def zipDir (dirpath, outFullName) :
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile (outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk (dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace (dirpath, '')
 
        for filename in filenames:
            zip.write (os.path.join (path, filename), os.path.join (fpath, filename))
    zip.close ()

data_transforms = {
    "train": A.Compose ([
        # 固定图像尺寸
        A.Resize (*CFG.img_size, interpolation = cv2.INTER_NEAREST),
        # 随机水平翻转
        A.HorizontalFlip (p = 0.5), 
#         A.VerticalFlip(p=0.5),
        # 随机平移、缩放、旋转
        A.ShiftScaleRotate (shift_limit = 0.0625, scale_limit = 0.05, rotate_limit = 10, p = 0.5), 
        A.OneOf ([
            # 以下两种操作以 p 的概率随机选择其中一种
            # 网格畸变
            A.GridDistortion (num_steps = 5, distort_limit = 0.05, p = 1.0), 
#             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
            # 弹性变换, 扭曲图像的同时保持图像的连续性
            A.ElasticTransform (alpha = 1, sigma = 50, alpha_affine = 50, p = 1.0)
#             A.ElasticTransform (alpha = 20, sigma = 5, alpha_affine = 20, p = 1.0)
        ], p = 0.25),
        # 对图片进行随机遮挡, 遮挡区域用固定值或者随机值填充
        A.CoarseDropout (max_holes = 8, max_height = CFG.img_size[0] // 20, 
                        max_width = CFG.img_size[1] // 20, min_holes = 5, 
                        fill_value = 0, mask_fill_value = 0, p = 0.5),
        ], p = 1.0),
    
    "valid": A.Compose ([
        A.Resize (*CFG.img_size, interpolation = cv2.INTER_NEAREST),
        ], p = 1.0)
}