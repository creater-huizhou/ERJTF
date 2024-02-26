import os
import csv
import random
import numpy as np


def fill_list(list, length):
    if length <= len(list):
        return list[:length]
    else:
        while len(list) < length:
            random_element = random.choice(list)
            list.append(random_element)
        return list


# 建立csv文件保存训练集和测试集路径
def creat_train_csv(input_directory, img_dir, mask_dir, output_directory, csv_name):
    # 如果csv文件不存在
    if not os.path.exists(os.path.join(input_directory, csv_name)):
        synthetic_crack_image_list = []
        # 遍历文件夹，获得所有的图片的路径
        synthetic_crack_img_path = os.path.join(input_directory, 'synthetic_crack', img_dir)
        for name in os.listdir(synthetic_crack_img_path):
            synthetic_crack_image_list.append(os.path.join(synthetic_crack_img_path, name))
        # 随机打散顺序
        np.random.shuffle(synthetic_crack_image_list)

        real_crack_image_list = []
        real_crack_img_path = os.path.join(input_directory, 'real_crack', img_dir)
        for name in os.listdir(real_crack_img_path):
            real_crack_image_list.append(os.path.join(real_crack_img_path, name))
        real_crack_image_list = fill_list(real_crack_image_list, len(synthetic_crack_image_list))
        # 随机打散顺序
        np.random.shuffle(real_crack_image_list)

        # 创建csv文件，并存储图片路径及其label信息
        with open(os.path.join(output_directory, csv_name), mode='w', newline='')as f:
            writer = csv.writer(f)
            i = 0
            for synthetic_crack_image in synthetic_crack_image_list:
                label_name = str(np.random.randint(1, 200)) + '.jpg'
                label = os.path.join(input_directory, 'synthetic_crack', mask_dir, label_name)
                writer.writerow([synthetic_crack_image, real_crack_image_list[i], label])
                i += 1
            print('written into csv file:', csv_name)


def creat_test_csv(input_directory, img_dir, mask_dir, output_directory, csv_name):
    # 如果csv文件不存在
    if not os.path.exists(os.path.join(input_directory, csv_name)):
        image_list = []
        # 遍历文件夹，获得所有的图片的路径
        for name in os.listdir(os.path.join(input_directory, img_dir)):
            image_list.append(os.path.join(input_directory, img_dir, name))

        # 随机打散顺序
        np.random.shuffle(image_list)
        # 创建csv文件，并存储图片路径及其label信息
        with open(os.path.join(output_directory, csv_name), mode='w', newline='')as f:
            writer = csv.writer(f)
            for img in image_list:
                name = img.split('/')[-1]
                label = os.path.join(input_directory, mask_dir, name)
                writer.writerow([img, label])
            print('written into csv file:', csv_name)