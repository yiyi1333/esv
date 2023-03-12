import pickle

import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image, ImageOps
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms


class SigComp2011_Dataset_Chinese(Dataset):
    def __init__(self, images, labels, transform=None, train=True):
        self.labels = labels
        self.images = images
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_transform(self):
        return self.transform


def build(genuine_num, forgery_num):
    # 数据路径
    images_dir_forgery = '../images/Sig2011/Forgery'
    images_dir_genuine = '../images/Sig2011/Genuine'
    # 读取数据, 生成数据集存储到本地
    # 真实签名
    image_list = os.listdir(images_dir_genuine)

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    genuine_map = {}
    for filename in image_list:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            num = filename.split('_')[0]
            # 使用 Image.open() 方法打开图片并将其添加到列表中
            image = Image.open(os.path.join(images_dir_genuine, filename))
            if (num not in genuine_map):
                genuine_map[num] = 1
            else:
                genuine_map[num] += 1

            # Image，以原图为中心，填充为1274 * 1274，使用白色填充
            image = ImageOps.pad(image, (1300, 1300), color=(255, 255, 255))
            # resize 为224 * 224
            image = image.resize((224, 224))

            if(genuine_map[num] <= genuine_num):
                train_images.append(image)
                train_labels.append((int(num) - 1) * 2 + 1)
            else:
                test_images.append(image)
                test_labels.append((int(num) - 1) * 2 + 1)

            print(filename, 'label:', (int(num) - 1) * 2 + 1)

    # 伪造签名
    image_list = os.listdir(images_dir_forgery)
    forgery_map = {}
    for filename in image_list:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            num = filename.split('_')[0][-3:]
            # 取字符串label的最后三个字符
            image = Image.open(os.path.join(images_dir_forgery, filename))
            image = ImageOps.pad(image, (1300, 1300), color=(255, 255, 255))
            # resize 为224 * 224
            image = image.resize((224, 224))

            if (num not in forgery_map):
                forgery_map[num] = 1
            else:
                forgery_map[num] += 1
            if(forgery_map[num] <= forgery_num):
                train_images.append(image)
                train_labels.append((int(num) - 1) * 2 + 2)
            else:
                test_images.append(image)
                test_labels.append((int(num) - 1) * 2 + 2)
            print(filename, 'label:', (int(num) - 1) * 2 + 2)

    # 生成数据集
    train_dataset = SigComp2011_Dataset_Chinese(train_images, train_labels, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]), True)
    test_dataset = SigComp2011_Dataset_Chinese(test_images, test_labels, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]), False)

    # 保存数据
    with open("sigComp2011_train_dataset_chinese.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    with open("sigComp2011_test_dataset_chinese.pkl", "wb") as f:
        pickle.dump(test_dataset, f)

def load(data_dir, train = True):
    if(train):
        with open(data_dir + 'sigComp2011_train_dataset_chinese.pkl', "rb") as f:
            train_dataset = pickle.load(f)
        return train_dataset
    else:
        with open(data_dir + '/sigComp2011_test_dataset_chinese.pkl', "rb") as f:
            test_dataset = pickle.load(f)
        return test_dataset

if __name__ == '__main__':
    build(17, 20)
    # train_dataset = load('', True)
    # print(train_dataset.get_transform())
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    # for i, data in enumerate(train_dataloader, 0):
    #     inputs, labels = data
    #     print(inputs)
    #     print(labels.shape)
    #     print(labels)
    #     break

# ToDo: target 归一化
# ToDo: transform
