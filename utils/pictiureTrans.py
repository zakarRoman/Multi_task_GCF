from PIL import Image
import torchvision.transforms as transforms
import os


def get_transform():
    theList = [transforms.ToTensor()]
    return transforms.Compose(theList)

def pic_trans(file_path):
    picTensorList = []
    theTranse = get_transform()
    for file_name in os.listdir(file_path):
        img = Image.open(file_path + "/" + file_name)
        img = theTranse(img)
        picTensorList.append(img)
    if len(picTensorList) == 1:
        return picTensorList[0]
    else:
        return picTensorList
