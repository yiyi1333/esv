import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

from net.VGG_16 import VGG16_3
from dataset import dataset_build
from dataset.dataset_build import SigComp2011_Dataset_Chinese
# forgery_path = 'images/Sig2011/Forgery'
# genuine_path = 'images/Sig2011/Genuine'
#
# forgery_list = os.listdir(forgery_path)
# genuine_list = os.listdir(genuine_path)

train_data = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

for data in train_loader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)

# 打开0.jpg
# image = Image.open('src/0.jpg')
# np_image = np.array(image)
# image = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5),
#                          (0.5, 0.5, 0.5))
# ])(image)