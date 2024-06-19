from model import EncoderDecoder, GCFDataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


def theTrans():
    return transforms.Compose([transforms.ToTensor()])


if __name__ == '__main__':
    trsft = theTrans()
    image = Image.open("1_pic.jpg")
    image_tensor = trsft(image).float()
    image_tensor = image_tensor.unsqueeze(0)
    print(image_tensor.shape)

    model = EncoderDecoder()
    output = model(image_tensor)
    print(f"结果是{output}")

    gcf_images = [...]
    labelList = [...]

    # 划分训练集和验证集
    train_images, val_images, train_labels, val_labels = train_test_split(gcf_images, labelList, test_size=0.2,
                                                                          random_state=42)

    # 创建数据集和数据加载器
    train_dataset = GCFDataset(train_images, train_labels)
    val_dataset = GCFDataset(val_images, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = EncoderDecoder()
    criterion = nn.MSELoss()  # 这里假设是回归问题，使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练过程
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # 验证过程
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

    print("Training complete.")
