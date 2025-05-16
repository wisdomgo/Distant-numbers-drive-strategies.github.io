import torch
import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.nn.functional as Func
import pandas as pd
from torch.utils.data import DataLoader
from utils.graph import *
from utils.siScore_utils import *
from utils.parameters import *
import os
from itertools import permutations
import copy
import timm
from timm.models.registry import register_model
from timm.models.vision_transformer import VisionTransformer

def create_vit_patch7_343(pretrained=False):
    """
    创建一个自定义的 Vision Transformer (ViT) 模型，适配 343x343 的输入图像和 7x7 的补丁大小。
    
    参数:
        pretrained (bool): 是否加载预训练权重。
    
    返回:
        VisionTransformer: 定制的 ViT 模型实例。
    """
    model = VisionTransformer(
        img_size=343,
        patch_size=7,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
    )
    if pretrained:
        # 预训练权重通常是基于特定的 img_size 和 patch_size
        # 需要额外处理以适配自定义模型
        raise NotImplementedError("预训练权重的调整需要额外处理。")
    return model

@register_model
def vit_base_patch7_343(pretrained=False, **kwargs):
    """
    注册自定义的 ViT 模型到 timm 的模型注册表中。
    
    参数:
        pretrained (bool): 是否加载预训练权重。
        **kwargs: 其他可选参数。
    
    返回:
        VisionTransformer: 注册的 ViT 模型实例。
    """
    return create_vit_patch7_343(pretrained=pretrained)

class betaSigmoid(nn.Module):
    def __init__(self, beta=5):
        super().__init__()
        self.beta = beta
    
    def forward(self, x):
        # 使用更陡峭的函数替代普通sigmoid
        return 1 / (1 + torch.exp(-self.beta * x))

class DistributionAlignmentLoss(nn.Module):
    def __init__(self, num_bins=10):
        super().__init__()
        self.num_bins = num_bins
        
    def forward(self, pred, target):
        # 分布对齐损失
        pred_hist = torch.histc(pred, bins=self.num_bins, min=0, max=1)
        target_hist = torch.histc(target, bins=self.num_bins, min=0, max=1)
        
        # 标准化直方图
        pred_hist = pred_hist / pred_hist.sum()
        target_hist = target_hist / target_hist.sum()
        
        pred_hist = pred_hist + 1e-8
        target_hist = target_hist + 1e-8

        # KL散度
        # try:
        #     distribution_loss = Func.kl_div(pred_hist.log(), target_hist, reduction='batchmean')
        # except:
        #     raise ValueError(f'pred_hist: {pred_hist}, target_hist: {target_hist}')
        
        distribution_loss = Func.kl_div(pred_hist.log(), target_hist, reduction='batchmean')
        return distribution_loss

class BoundaryAwareLoss(nn.Module):
    def __init__(self, boundary_weight=2.0):
        super().__init__()
        self.boundary_weight = boundary_weight
        
    def forward(self, pred, target):
        
        # 对接近0和1的目标值给予更大的权重
        boundary_mask = (target < 0.2) | (target > 0.8)
        boundary_loss = Func.mse_loss(
            pred[boundary_mask], 
            target[boundary_mask]
        ) if boundary_mask.any() else 0
        
        return self.boundary_weight * boundary_loss

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.distribution_loss = DistributionAlignmentLoss()
        self.boundary_loss = BoundaryAwareLoss()
        
    def forward(self, pred, target):
        base_loss = Func.mse_loss(pred, target)
        return base_loss + self.distribution_loss(pred, target) + self.boundary_loss(pred, target)
    
class BackboneModel(nn.Module):
    def __init__(self, pretrain=False, num_gpus=1, output_dim=1):
        super(BackboneModel, self).__init__()
        if pretrain:
            print('Loading the pre-trained ResNet.')
            self.model = models.resnet18(weights='DEFAULT')
        else:
            print('Training the ResNet from scratch.')
            self.model = models.resnet18(weights=None)

        input_dim = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
        self.activation = betaSigmoid(beta=5)

        self.num_gpus = num_gpus
        if self.num_gpus > 1 and torch.cuda.device_count() >= self.num_gpus:
            self.model = nn.DataParallel(self.model, device_ids=list(range(self.num_gpus)))
            cudnn.benchmark = True
            print(f"Using {self.num_gpus} GPUs for training.")
        else:
            print("Using a single GPU or CPU for training.")

    def initialize_weights(self):
        print('Initializing the weight of the model.')
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.model(x)
        out = self.activation(out)
        return out

class BackboneModel2(nn.Module):
    def __init__(self, num_gpus=1, output_dim=1, vit_model='vit_base_patch7_343'):
        """
        BackboneModel 使用 Vision Transformer (ViT) 作为骨架模型，并适配343x343的输入图片。

        参数:
            pretrain (bool): 是否使用预训练的权重。
            num_gpus (int): 使用的 GPU 数量。
            output_dim (int): 模型的输出维度。
            vit_model (str): 使用的 ViT 模型名称（例如 'vit_base_patch7_343'）。
        """
        super(BackboneModel2, self).__init__()

        self.model = timm.create_model(vit_model, pretrained=False)
        
        # 获取 ViT 模型分类头的输入特征数
        input_dim = self.model.head.in_features

        # 替换分类头
        self.model.head = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

        # 自定义激活函数
        self.activation = betaSigmoid(beta=5)

        self.num_gpus = num_gpus
        if self.num_gpus > 1 and torch.cuda.device_count() >= self.num_gpus:
            self.model = nn.DataParallel(self.model, device_ids=list(range(self.num_gpus)))
            cudnn.benchmark = True
            print(f"Using {self.num_gpus} GPUs for training.")
        else:
            print("Using a single GPU or CPU for training.")
    
    def initialize_weights(self):
        """
        初始化模型权重。注意，预训练模型的权重通常已经初始化，
        仅替换的分类头可能需要初始化。
        """
        print('Initializing the weight of the model.')
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量。
        """
        out = self.model(x)
        out = self.activation(out)
        return out