import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 构建Resnet块
# class ResnetBlock(nn.module):
#     def __init__(self):

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

    # 使用编码器-解码器架构
class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(),
            nn.Tanh(),
            nn.Conv2d(),
            nn.ReLU(),
            nn.Conv2d(),
            nn.ReLU(),
            nn.MaxPool2d()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.fc = nn.Linear(1 * 64 * 64, 2)  # 假设输入图片为128x128

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        flat = decoded.view(decoded.size(0), -1)
        output = self.fc(flat)
        return output