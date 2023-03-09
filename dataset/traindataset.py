import pickle

import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


class SigComp2011_Train_Dataset_Chinese(Dataset):
    def __init__(self, images, labels):
        self.labels = labels
        self.images = images


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return torch.tensor(image), torch.tensor(label)

def get_data():
#     判断某个文件是否存在
    if os.path.exists("sigComp2011_train_dataset_chinese.pkl"):
        with open("sigComp2011_train_dataset_chinese.pkl", "rb") as f:
            dataset = pickle.load(f)
    else:
        # 读取数据, 生成数据集存储到本地
        # 真实签名
        image_dir = '../images/trainingSet/Chinese/TrainingSet/Offline_Genuine'
        image_list = os.listdir(image_dir)
        # image_list = sorted(image_list, key=lambda x: (int(x.split('_')[0]), int(x.split('.')[0].split('_')[1])))
        images = []
        labels = []
        for filename in image_list:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # 使用 Image.open() 方法打开图片并将其添加到列表中
                image = Image.open(os.path.join(image_dir, filename))
                images.append(image)
                # 生成标签，转为数字
                label = filename.split('_')[0]
                labels.append((int(label) - 1) * 2 + 1)
                print(filename, 'label:', (int(label) - 1) * 2 + 1)

        # 伪造签名
        image_dir = '../images/trainingSet/Chinese/TrainingSet/Offline_Forgeries'
        image_list = os.listdir(image_dir)
        for filename in image_list:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image = Image.open(os.path.join(image_dir, filename))
                images.append(image)
                label = filename.split('_')[0][-3:]
                # 取字符串label的最后三个字符
                labels.append((int(label) - 1) * 2 + 2)
                print(filename, 'label:', (int(label) * 2))

        dataset = SigComp2011_Train_Dataset_Chinese(images, labels)
        with open("sigComp2011_train_dataset_chinese.pkl", "wb") as f:
            pickle.dump(dataset, f)
    return dataset

dataset = get_data()
my_dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)
print(len(my_dataloader))