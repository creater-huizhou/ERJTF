import random
import numpy as np
import torch
import torchvision.utils
from torch.autograd import Variable
import os
import csv
import datetime
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from model import *
from dataset import *
from generate_data_csv import *
from loss import *
from patch_position_embedding import PatchPositionEmbeddingSine


def train(model, max_epochs, batchs, save_dir_path, save_result_path, input_pos):
    model.train()

    min_train_loss = 10000
    epochs_without_improvement = 0
    patience = 3
    step_num = 0

    total_loss = 0
    total_rec_loss = 0
    total_seg_loss = 0
    total_consistency_loss = 0

    learning_rate = 0.0005
    interval = 400

    for epoch in range(1, max_epochs + 1):
        # 建立csv文件保存训练集路径
        input_path = './CRACK500-train'
        output_path = './CRACK500-train'
        creat_train_csv(input_path, 'train_imgs', 'train_masks', output_path, 'train_data.csv')
        # 准备数据集
        train_dataset = My_train_Dataset(output_path, 'train_data.csv')
        # 把数据集装载到DataLoader里
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batchs, drop_last=True)

        for step, (synthetic_crack_img, origin_img, mask, real_crack_img, img_path, origin_path, mask_path) in enumerate(train_loader):  # 遍历训练集
            step_num += 1
            current_time = datetime.datetime.now()
            if step_num == 1:
                print("Time: {}, iters: {:5d}, learning_rate: {}".format(str(current_time), step_num, learning_rate))
            if step_num % 2000 == 0:
                learning_rate *= 0.8
                print("Time: {}, iters: {:5d}, learning_rate: {}".format(str(current_time), step_num, learning_rate))
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

            synthetic_crack_img = Variable(synthetic_crack_img).cuda()
            origin_img = Variable(origin_img).cuda()
            mask = Variable(mask).cuda()
            input_pos = Variable(input_pos).cuda()
            real_crack_img = Variable(real_crack_img).cuda()

            # 前向计算获得重建的图片
            synthetic_crack_img_rec, synthetic_crack_img_residual, synthetic_crack_mask_pred = model(synthetic_crack_img, input_pos)
            real_crack_img_rec, real_crack_img_residual, real_crack_mask_pred = model(real_crack_img, input_pos)
            real_crack_mask_pred = real_crack_mask_pred.repeat(1, 3, 1, 1).detach()
            # random_noisy = nn.Parameter(random_noisy, requires_grad=True).cuda()
            random_noisy = (torch.randint(20, 80, size=(batchs, 3, 128, 128)).float() / torch.Tensor([255.0]) - torch.Tensor([0.5])) / torch.Tensor([0.5])
            random_noisy = random_noisy.cuda()
            real_crack_img_pred = real_crack_img_rec.detach() * (torch.Tensor([1.0]).cuda() - real_crack_mask_pred) + random_noisy * real_crack_mask_pred

            # 计算损失
            loss, rec_loss, seg_loss, consistency_loss = cal_loss(synthetic_crack_img_rec,
                                                                  origin_img,
                                                                  synthetic_crack_mask_pred,
                                                                  mask,
                                                                  real_crack_img_pred,
                                                                  real_crack_img)

            total_loss += loss.item()
            total_rec_loss += rec_loss.item()
            total_seg_loss += seg_loss.item()
            total_consistency_loss += consistency_loss.item()

            if step_num % interval == 0:
                if not os.path.exists(os.path.join(save_result_path, 'save_mid')):
                    os.mkdir(os.path.join(save_result_path, 'save_mid'))
                name1 = os.path.join(save_result_path, 'save_mid', str(step_num) + '_a_synthetic_img.png')
                torchvision.utils.save_image(synthetic_crack_img, name1, normalize=True)
                name2 = os.path.join(save_result_path, 'save_mid', str(step_num) + '_b_orginal_img.png')
                torchvision.utils.save_image(origin_img, name2, normalize=True)
                name3 = os.path.join(save_result_path, 'save_mid', str(step_num) + '_c_synthetic_img_rec.png')
                torchvision.utils.save_image(synthetic_crack_img_rec, name3, normalize=True)
                name4 = os.path.join(save_result_path, 'save_mid', str(step_num) + '_d_synthetic_img_residual.png')
                torchvision.utils.save_image(synthetic_crack_img_residual, name4, normalize=True)
                name5 = os.path.join(save_result_path, 'save_mid', str(step_num) + '_e_synthetic_img_mask.png')
                torchvision.utils.save_image(mask, name5, normalize=True)
                name6 = os.path.join(save_result_path, 'save_mid', str(step_num) + '_f_mask_pred.png')
                torchvision.utils.save_image(synthetic_crack_mask_pred, name6, normalize=True)

                name7 = os.path.join(save_result_path, 'save_mid', str(step_num) + '_g_real_crack_img.png')
                torchvision.utils.save_image(real_crack_img, name7, normalize=True)
                name8 = os.path.join(save_result_path, 'save_mid', str(step_num) + '_h_real_crack_img_rec.png')
                torchvision.utils.save_image(real_crack_img_rec, name8, normalize=True)
                name9 = os.path.join(save_result_path, 'save_mid', str(step_num) + '_i_real_crack_img_residual.png')
                torchvision.utils.save_image(real_crack_img_residual, name9, normalize=True)
                name10 = os.path.join(save_result_path, 'save_mid', str(step_num) + '_j_real_crack_mask_pred.png')
                torchvision.utils.save_image(real_crack_mask_pred, name10, normalize=True)
                name11 = os.path.join(save_result_path, 'save_mid', str(step_num) + '_k_real_crack_img_pred.png')
                torchvision.utils.save_image(real_crack_img_pred, name11, normalize=True)

                train_loss = total_loss / interval
                train_rec_loss = total_rec_loss / interval
                train_seg_loss = total_seg_loss / interval
                train_consistency_loss = total_consistency_loss / interval
                print("iters: {:>5d}, total_loss: {:.6f}, reconstruction_loss: {:.6f}, "
                      "segmentation_loss: {:.6f}, consistency_loss: {:.6f}".format(step_num,
                                                                                   train_loss,
                                                                                   train_rec_loss,
                                                                                   train_seg_loss,
                                                                                   train_consistency_loss))

                # 保存loss数据到csv文件
                item = {'iters': str(step_num),
                        'total_loss': str(train_loss),
                        'reconstruction_loss': str(train_rec_loss),
                        'segmentation_loss': str(train_seg_loss),
                        'consistency_loss': str(train_consistency_loss)
                        }
                fieldnames = ['iters', 'total_loss', 'reconstruction_loss', 'segmentation_loss', 'consistency_loss']
                save_loss = os.path.join(save_result_path, 'loss.csv')
                with open(save_loss, mode='a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    # 判断表格内容是否为空，如果为空就添加表头
                    if not os.path.getsize(save_loss):
                        writer.writeheader()  # 写入表头
                    writer.writerows([item])

                if train_loss < min_train_loss:
                    # 模型保存
                    # save_para_name = 'model-epoch-' + str(epoch) + '.pt'
                    save_para_name = 'model-iters-' + str(step_num) + '.pt'
                    save_para_path = os.path.join(save_dir_path, save_para_name)
                    torch.save({'params': model.state_dict()}, save_para_path)
                    min_train_loss = train_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement = epochs_without_improvement + 1

                # 如果验证集上的损失连续patience个epoch没有提高，则停止训练
                if epochs_without_improvement == patience:
                    # print('Early stopping at epoch {}...'.format(epoch+1))
                    print('Early stopping at iters {}...'.format(step_num))
                    return
                if epochs_without_improvement == 1 or epochs_without_improvement == 2:
                    # print('Early stopping at epoch {}...'.format(epoch+1))
                    learning_rate *= 0.8
                    print("Time: {}, iters: {:5d}, model's performence not improves, learning_rate changes to {}".format(
                            str(current_time), step_num, learning_rate))


                total_loss = 0
                total_rec_loss = 0
                total_seg_loss = 0
                total_consistency_loss = 0

            # 更新梯度等参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICE'] = '0, 1'
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    batch_size = 32
    max_epochs = 20

    save_dir_path = 'CRACK500-train/model_save'
    save_result_path = 'CRACK500-train'
    # 如果文件夹不存在，就创建文件夹
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)

    # 查看GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_pos = PatchPositionEmbeddingSine(ksize=4, stride=4)
    input_pos = input_pos.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    input_pos = input_pos.flatten(2).permute(0, 2, 1)
    # print(input_pos.shape)

    model = nn.DataParallel(CMT_TransCNN(), device_ids=[0, 1]).to(device)
    # model_para = torch.load('CRACK500-train/model_save/model-iters-6800.pt')
    # model.load_state_dict(model_para['params'])
    train(model, max_epochs, batch_size, save_dir_path, save_result_path, input_pos)