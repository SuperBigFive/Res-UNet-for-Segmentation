from config import CFG
import torch
import segmentation_models_pytorch as smp

def build_model (backbone = CFG.backbone):
    model = smp.Unet (
        # 更换成 resnet50
        encoder_name = backbone, 
        # 定义编码器网络深度, 设为 4，加速训练过程
#         encoder_depth = 4,
        # 不设预训练权重, 从零开始训练!
        encoder_weights = None,
        # 没有 decoder_name, 由模型框架 Unet 决定
        # 给定图片是单通道的, 读入时叠加成三通道
        in_channels = 3, 
        # 分类类别参数, 对于 UWMGI 数据集是 3
        classes = CFG.num_classes,
        activation = None,
#         segmentation_head = segmentation_head,
    )
    model.to (CFG.device)
    return model

def load_model (backbone, path) :
    '''
        用于测试模型前加载训练过程中表现最好的模型
    '''
    model = build_model (backbone)
    model.load_state_dict (torch.load (path, map_location = torch.device('cpu')))
    model.eval ()
    return model