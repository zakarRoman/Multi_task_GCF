from PIL import Image
import torchvision.transforms as transforms
import os
import torch


def get_transform():
    theList = [transforms.ToTensor()]
    return transforms.Compose(theList)


def pic_trans(file_path):
    picTensorList = []
    theTrans = get_transform()
    for file_name in os.listdir(file_path):
        img = Image.open(file_path + "/" + file_name)
        img = theTrans(img)
        picTensorList.append(img)
    if len(picTensorList) == 1:
        picTensor = torch.Tensor(picTensorList[0])
        picTensor = picTensor.unsqueeze(0)
        return picTensor
    else:
        stacked_tensor = torch.stack(picTensorList, dim=0)
        return stacked_tensor
