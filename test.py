
import torch
from torch.utils.data import DataLoader
from torchvision import utils
import cv2
import os
from PIL import Image
import numpy as np
from model import *
from dataset import *
from generate_data_csv import *
from patch_position_embedding import PatchPositionEmbeddingSine
from cal_mask_OIS import cal_mask_ois
from cal_mask_ODS import cal_mask_ods
from cal_mask_AP import  cal_mask_ap
import datetime

    
# 定义图像拼接函数
def concat_image(number, row, col, path1, path2):
    to_image = Image.new('RGB', (col * 128, row * 128))  # 创建一个新图
    count = 0
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, row + 1):
        for x in range(1, col + 1):
            count = count + 1
            name = str(number) + '_' + str(count) + '.jpg'
            from_image = Image.open(os.path.join(path1, name))
            to_image.paste(from_image, ((x - 1) * 128, (y - 1) * 128))

    to_image.save(os.path.join(path2, str(number) + '.jpg'))  # 保存新图
    

def generate_norm_map(ori_img, img_pred, img_residual, mask_pred, img_paths, result_path, residual_path, mask_pred_path):
    # 如果文件夹不存在，就创建文件夹
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(residual_path):
        os.mkdir(residual_path)
    if not os.path.exists(mask_pred_path):
        os.mkdir(mask_pred_path)
    imgname = img_paths.split('/')[-1]
    name = imgname.split('.')[0]
    # 原图
    path1 = name + '_a.png'
    input_img = (ori_img + 1) / 2
    utils.save_image(input_img, os.path.join(result_path, path1))
    # 预测图像
    path2 = name + '_b.png'
    output_img = (img_pred + 1) / 2
    utils.save_image(output_img, os.path.join(result_path, path2))
    # 异常图
    path3 = name + '_c.png'
    # norm = (img_residual + 1) / 2
    norm = img_pred - ori_img
    utils.save_image(norm, os.path.join(result_path, path3))

    path4 = name + '_d.png'
    utils.save_image(mask_pred, os.path.join(result_path, path4))

    utils.save_image(norm, os.path.join(residual_path, imgname))
    utils.save_image(mask_pred, os.path.join(mask_pred_path, imgname))



def test(result_path, residual_path, mask_pred_path, input_pos, batch_size):
    if not os.path.exists(result_path):
        for step, (img, mask, img_path, mask_path) in enumerate(test_loader):
            img = img.to(device)
            input_pos = input_pos.to(device)
            # 前向计算获得重建的图片
            img_pred, img_residual, mask_pred = model(img, input_pos)
            for i in range(batch_size):
                img_preds = img_pred[i].squeeze(0)
                img_residuals = img_residual[i].squeeze(0)
                mask_preds = mask_pred[i].squeeze(0)
                ori_imgs = img[i].squeeze(0)
                img_paths = img_path[i]
                generate_norm_map(ori_imgs, img_preds, img_residuals, mask_preds, img_paths, result_path, residual_path, mask_pred_path)



# 建立csv文件保存训练集和测试集路径
input_output_path = 'CRACK500-test'
creat_test_csv(input_output_path, 'test_imgs', 'test_masks', input_output_path, 'test_data.csv')
if input_output_path == 'AEL-test':
    batch_size = 16
elif input_output_path == 'CFD-test' or input_output_path == 'CRACK500-test':
    batch_size = 4
else:
    batch_size = 1

# 准备数据集
test_dataset = My_test_Dataset(input_output_path, 'test_data.csv')
# 把数据集装载到DataLoader里
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
# 查看GPU是否可用
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(CMT_TransCNN(), device_ids=[0, 1]).to(device)
    
#list = ['4000', '4400', '4800', '5200', '5600', '6000', '6400', '6800', '7200', '7600', '8000', '8400', '8800', '9200', '9600', '10000']
list = ['4000']

for idx in list:
    iter_num = 'iters-' + idx
    print(iter_num)
    model_para = torch.load('CRACK500-train/model_save/model-' + iter_num + '.pt')
    model.load_state_dict(model_para['params'])
    model.eval()

    res_dir_1 = input_output_path + '/result'
    if not os.path.exists(res_dir_1):
        os.mkdir(res_dir_1)
    res_dir = os.path.join(res_dir_1, iter_num)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    input_pos = PatchPositionEmbeddingSine(ksize=4, stride=4)
    input_pos = input_pos.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # self.input_pos为[4, 256, 64, 64]
    input_pos = input_pos.flatten(2).permute(0, 2, 1)  # self.input_pos为[4, 4096, 256]

    # 测试输出图像保存路径
    result_path = res_dir + '/results'
    # 预残差图保存路径
    residual_path = result_path + '-residual'
    mask_pred_path = result_path + '-mask'
    current_time_1 = datetime.datetime.now()
    test(result_path, residual_path, mask_pred_path, input_pos, batch_size)
    current_time_2 = datetime.datetime.now()
    print("Inference time:{}".format(current_time_2 - current_time_1))
    
    """
    mask_path = input_output_path + '/masks_resize'
    residual_concat_path = result_path + '-residual-concat' # 拼接后图片地址
    mask_pred_concat_path =  result_path + '-mask-pred-concat'
    if not os.path.exists(residual_concat_path):
        os.mkdir(residual_concat_path)
    if not os.path.exists(mask_pred_concat_path):
        os.mkdir(mask_pred_concat_path)
    
    for img_name in os.listdir(mask_path):
        # print(img_name)
        image = np.array(Image.open(os.path.join(mask_path, img_name)).convert("L"))
        h = image.shape[0]
        w = image.shape[1]
        row = h // 128
        col = w // 128
        number = img_name.split('.')[0]
        concat_image(number, row, col, residual_path, residual_concat_path)
        concat_image(number, row, col, mask_pred_path, mask_pred_concat_path)
    """
    
    # cal_mask_ois(input_output_path, res_dir)
    # cal_mask_ods(input_output_path, res_dir)
    # cal_mask_ap(input_output_path, res_dir)