import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


# 构建Resnet块
class ResnetBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += residual
        out = self.relu(out)
        return out


class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),  # 输入通道数为3，假设输入图片为RGB图像
            nn.InstanceNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 288x360 -> 144x180
            ResnetBlock(128),
            ResnetBlock(128),
            ResnetBlock(128),
            ResnetBlock(128),
            ResnetBlock(128),
            ResnetBlock(128)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 144x180 -> 288x360
            nn.ReLU(inplace=True),
            ResnetBlock(64),
            ResnetBlock(64),
            ResnetBlock(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  # 288x360 -> 288x360
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(3),
            nn.Sigmoid()  # 输出为定位图像，使用Sigmoid激活函数
        )

        # 假设解码器最后输出的特征图大小为3x288x360（输入图片大小）
        self.fc = nn.Linear(3 * 288 * 360, 2)  # 这里假设扁平化后的向量长度为3 * 288 * 360

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        flat = decoded.view(decoded.size(0), -1)
        output = self.fc(flat)
        return output


# 定义DataSet来加载图片
class GCFDataset(Dataset):
    def __init__(self, gcf_images, labels):
        self.gcf_images = gcf_images
        self.labels = labels

    def __len__(self):
        return len(self.gcf_images)

    def __getitem__(self, idx):
        image = self.gcf_images[idx]
        label = self.labelsp[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
