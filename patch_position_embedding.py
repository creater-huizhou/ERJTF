import torch
import torchvision.utils
from torch.utils.data import DataLoader
from torchvision import utils
import cv2
import torch
import os
import numpy as np


# 正余弦位置编码
def PatchPositionEmbeddingSine(ksize, stride):
    temperature = 10000.0
    feature_h = int((128.0 - ksize) / stride) + 1 # 64
    num_pos_feats = 128.0 # 128
    mask = torch.ones((feature_h, feature_h))
    # 行方向求元素的累积和
    y_embed = mask.cumsum(0, dtype=torch.float32)
    # 列方向求元素的累积和
    x_embed = mask.cumsum(1, dtype=torch.float32)
    # 产生一维张量[0, 1, 2, ..., num_pos_feats-1]
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = torch.div(dim_t, 2, rounding_mode='floor')
    dim_t = temperature ** (2 * dim_t / num_pos_feats)

    pos_x = x_embed[:, :, None] / dim_t # x_embed[:, :, None]为[64, 64, 1], dim_t为[128], pos_x为[64, 64, 128]
    pos_y = y_embed[:, :, None] / dim_t # 同上
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1) # pos为[256, 64, 64]

    return pos