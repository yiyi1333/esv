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
    def __init__(self, images, labels, train=True, transform=None):
        self.labels = labels
        self.train = train
        self.transform= transform
        self.images = images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


def to_grayscale(img):
    return torch.mean(img, dim=0, keepdim=True)

def build(genuine_num, forgery_num):
    # 图像处理方法
    tran = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.GaussianBlur(3, (0.1, 2.0))
        ])
    # 数据路径
    # images_dir_forgery = '../images/Sig2011/Forgery'
    # images_dir_genuine = '../images/Sig2011/Genuine'
    images_dir_forgery = '../images/Sig2011_enhancement/Forgery'
    images_dir_genuine = '../images/Sig2011_enhancement/Genuine'
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
            # 过滤num > 005 的数据
            if (int(num) > 5):
                continue

            # 使用 Image.open() 方法打开图片并将其添加到列表中
            image = Image.open(os.path.join(images_dir_genuine, filename))
            if (num not in genuine_map):
                genuine_map[num] = 1
            else:
                genuine_map[num] += 1

            # Image，以原图为中心，填充为正方形，使用白色填充
            edge = max(image.size[0], image.size[1])
            image = ImageOps.pad(image, (edge, edge), color=(255, 255, 255))
            # resize 为224 * 224 双线性插值方法

            image = image.resize((224, 224))
            # totensor
            image = tran(image)
            # 通道压缩
            # to_gray = transforms.Lambda(lambda x: to_grayscale(x))
            # image = to_gray(image)

            if(genuine_map[num] <= genuine_num):
                train_images.append(image)
                # train_labels.append((int(num) - 1) * 2)
                train_labels.append(int(num) - 1)
            else:
                test_images.append(image)
                # test_labels.append((int(num) - 1) * 2)
                test_labels.append(int(num) - 1)

            print(filename, 'label:', (int(num) - 1))

    # 伪造签名
    image_list = os.listdir(images_dir_forgery)
    forgery_map = {}
    for filename in image_list:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            num = filename.split('_')[0][-3:]

            # 过滤num > 005 的数据
            if (int(num) > 5):
                continue

            # 取字符串label的最后三个字符
            image = Image.open(os.path.join(images_dir_forgery, filename))
            # Image，以原图为中心，填充为正方形，使用白色填充
            edge = max(image.size[0], image.size[1])
            image = ImageOps.pad(image, (edge, edge), color=(255, 255, 255))
            # resize 为224 * 224
            image = image.resize((224, 224))
            # totensor
            image = tran(image)
            # 通道压缩
            # to_gray = transforms.Lambda(lambda x: to_grayscale(x))
            # image = to_gray(image)

            if (num not in forgery_map):
                forgery_map[num] = 1
            else:
                forgery_map[num] += 1
            if(forgery_map[num] <= forgery_num):
                train_images.append((image))
                # train_labels.append((int(num) - 1) * 2 + 1)
                train_labels.append(int(num) - 1)
            else:
                test_images.append((image))
                # test_labels.append((int(num) - 1) * 2 + 1)
                test_labels.append(int(num) - 1)
            print(filename, 'label:', (int(num) - 1))

    # 生成数据集
    train_dataset = SigComp2011_Dataset_Chinese(train_images, train_labels, True, transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.GaussianBlur(3, (0.1, 2.0))
        ]))
    test_dataset = SigComp2011_Dataset_Chinese(test_images, test_labels,  False, transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.GaussianBlur(3, (0.1, 2.0))
        ]))

    # 保存数据
    with open("sigComp2011_train_dataset_chinese.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    with open("sigComp2011_test_dataset_chinese.pkl", "wb") as f:
        pickle.dump(test_dataset, f)

def load(data_dir, train = True):
    if(train):
        with open(data_dir + '/sigComp2011_train_dataset_chinese.pkl', "rb") as f:
            train_dataset = pickle.load(f)
        return train_dataset
    else:
        with open(data_dir + '/sigComp2011_test_dataset_chinese.pkl', "rb") as f:
            test_dataset = pickle.load(f)
        return test_dataset


