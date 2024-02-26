import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import os
import csv
import numpy as np
import cv2


# 从csv文件中读取训练集和测试集路径信息
def load_train_data_csv(root, csv_name):
    synthetic_crack_image_list, real_crack_image_list, label_list = [], [], []
    with open(os.path.join(root, csv_name)) as f:
        reader = csv.reader(f)
        for row in reader:
            synthetic_crack_image, real_crack_image, label = row
            synthetic_crack_image_list.append(synthetic_crack_image)
            real_crack_image_list.append(real_crack_image)
            label_list.append(label)
    # 返回图片路径list和标签list
    return synthetic_crack_image_list, real_crack_image_list, label_list


def load_test_data_csv(root, csv_name):
    image_list, label_list = [], []
    with open(os.path.join(root, csv_name)) as f:
        reader = csv.reader(f)
        for row in reader:
            img, label = row
            image_list.append(img)
            label_list.append(label)
    # 返回图片路径list和标签list
    return image_list, label_list


# 合成图像
def composite_image(img, mask):
    # 读入原始图片
    # origin = np.array(img).transpose((1, 2, 0))
    origin = np.array(img)
    # 随机生成噪声图片
    noisy = np.random.randint(20, 80, size=[128, 128])
    noisy = noisy[:, :, np.newaxis]
    noisy = noisy.repeat(3, axis=2)
    mask = np.array(mask) / 255.
    mask = mask[:, :, np.newaxis]
    mask = mask.repeat(3, axis=2)
    # 合成图片
    composite = origin * (1 - mask) + noisy * mask
    composite = cv2.blur(composite, (3, 3))

    return Image.fromarray(np.uint8(composite)).convert('RGB')


# 自定义数据集
class My_train_Dataset(Dataset):
    def __init__(self, data_path, csv_name):
        # self.data_list, self.origin_list, self.mask_list = load_train_data_csv(data_path, csv_name)
        self.synthetic_data_list, self.real_data_list, self.mask_list = load_train_data_csv(data_path, csv_name)
        self.data_transform = transforms.Compose([
            transforms.ColorJitter(brightness=[0.8, 1.2], contrast=[0.7, 1.3], saturation=[0.5, 1.5], hue=[-0.02, 0.02]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.origin_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        origin_img = Image.open(self.synthetic_data_list[index]).convert('RGB')
        mask = Image.open(self.mask_list[index]).convert('L')
        img = composite_image(origin_img, mask)
        img = self.data_transform(img)
        origin_img = self.origin_transform(origin_img)
        mask = self.mask_transform(mask)
        real_crack_img = Image.open(self.real_data_list[index]).convert('RGB')
        real_crack_img = self.data_transform(real_crack_img)

        return img, origin_img, mask, real_crack_img, self.synthetic_data_list[index], self.synthetic_data_list[index], self.mask_list[index]

    def __len__(self):
        return len(self.synthetic_data_list)


class My_test_Dataset(Dataset):
    def __init__(self, data_path, csv_name):
        self.data_list, self.mask_list = load_test_data_csv(data_path, csv_name)
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = Image.open(self.data_list[index]).convert('RGB')
        img = self.data_transform(img)
        mask = Image.open(self.mask_list[index]).convert('L')
        mask = self.mask_transform(mask)
        return img, mask, self.data_list[index], self.mask_list[index]

    def __len__(self):
        return len(self.data_list)