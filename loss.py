import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        perceptual_loss = 0
        style_loss = 0
        for i in range(len(x_vgg)):
            perceptual_loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        style_loss += self.criterion(self.compute_gram(x_vgg[3]), self.compute_gram(y_vgg[3]))
        style_loss += self.criterion(self.compute_gram(x_vgg[4]), self.compute_gram(y_vgg[4]))
        return perceptual_loss, style_loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

class MSSSIMLoss(torch.nn.Module):
    def __init__(self, size_average = True, max_val = 1):
        super(MSSSIMLoss, self).__init__()
        self.size_average = size_average
        self.channel = 3
        self.max_val = max_val
    def _ssim(self, img1, img2, size_average = True):
        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = create_window(window_size, sigma, self.channel).cuda()
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = self.channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = self.channel) - mu1_mu2

        C1 = (0.01*self.max_val)**2
        C2 = (0.03*self.max_val)**2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda())

        msssim = Variable(torch.Tensor(levels,).cuda())
        mcs = Variable(torch.Tensor(levels,).cuda())
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = torch.prod(mcs[0: levels-1] ** weight[0: levels-1]) * (msssim[levels-1] ** weight[levels-1])

        return value

    def forward(self, img1, img2):

        return 1 - self.ms_ssim(img1, img2)


# 计算loss
def cal_loss(img_pred, img_truth, mask_pred, mask, crack_img_pred, crack_img_truth):
    l1_loss = nn.L1Loss()
    content_loss = ContentLoss()
    ms_ssim_loss = MSSSIMLoss(max_val=1)
    seg_loss = nn.BCELoss(weight=torch.tensor([5]).cuda())
    # 损失权重
    L1_LOSS_WEIGHT = torch.tensor([10]).cuda()
    PERCEPTUAL_LOSS_WEIGHT = torch.tensor([10]).cuda()
    STYLE_LOSS_WEIGHT = torch.tensor([250]).cuda()
    MS_SSIM_LOSS_WEIGHT = torch.tensor([10]).cuda()
    SEG_LOSS_WEIGHT = torch.tensor([5]).cuda()
    CONSISTENCY_LOSS_WEIGHT = torch.tensor([0.1]).cuda()

    L1_Loss = l1_loss(img_pred, img_truth)
    Perceptual_Loss, Style_Loss = content_loss(img_pred, img_truth)
    MS_SSIM_Loss = ms_ssim_loss(torch.sigmoid(img_pred), torch.sigmoid(img_truth))

    crack_L1_Loss = l1_loss(crack_img_pred, crack_img_truth)
    crack_Perceptual_Loss, crack_Style_Loss = content_loss(crack_img_pred, crack_img_truth)
    crack_MS_SSIM_Loss = ms_ssim_loss(torch.sigmoid(crack_img_pred), torch.sigmoid(crack_img_truth))

    Reconstruction_Loss =  L1_LOSS_WEIGHT * L1_Loss + PERCEPTUAL_LOSS_WEIGHT * Perceptual_Loss + STYLE_LOSS_WEIGHT * Style_Loss + MS_SSIM_LOSS_WEIGHT * MS_SSIM_Loss

    Seg_Loss = SEG_LOSS_WEIGHT * seg_loss(mask_pred, mask)

    Consistency_Loss = CONSISTENCY_LOSS_WEIGHT * (L1_LOSS_WEIGHT * crack_L1_Loss + PERCEPTUAL_LOSS_WEIGHT * crack_Perceptual_Loss + STYLE_LOSS_WEIGHT * crack_Style_Loss) + MS_SSIM_LOSS_WEIGHT * crack_MS_SSIM_Loss

    loss = Reconstruction_Loss + Seg_Loss + Consistency_Loss

    return loss, Reconstruction_Loss, Seg_Loss, Consistency_Loss
