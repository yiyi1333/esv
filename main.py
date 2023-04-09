# import torchvision
#
# google_inception = torchvision.models.inception_v3(pretrained=True)
#
# print(google_inception)
from PIL import ImageOps, Image
import random

from torchvision import transforms


# import torch
# import torchvision.transforms as transforms
# from PIL import Image
#
# # 加载图像并将其转换为 PyTorch 张量
# img = Image.open('src/0.jpg')
# to_tensor = transforms.ToTensor()
# tensor_img = to_tensor(img)
#
# # 定义一个函数，该函数将三通道张量转换为单通道张量
# def to_grayscale(img):
#     return torch.mean(img, dim=0, keepdim=True)
#
# # 使用 Lambda 转换将三通道张量转换为单通道张量
# to_gray = transforms.Lambda(lambda x: to_grayscale(x))
# gray_tensor = to_gray(tensor_img)
#
# # 显示结果
# print('Original tensor shape:', tensor_img.shape)
# print('Grayscale tensor shape:', gray_tensor.shape)

def enhancement(img):
    # 将图片顺时针或逆时针随机旋转8度以内,使用白色填充
    img = img.rotate(random.randint(-8, 8), fillcolor=(254, 254, 254))
    # 将图片放大或缩小到原来的0.9~1.1倍
    img = img.resize((int(img.size[0] * random.uniform(0.9, 1.1)), int(img.size[1] * random.uniform(0.9, 1.1))))
    # 将图片随机平移0~40像素
    img = ImageOps.expand(img, border=random.randint(0, 40), fill=(254, 254, 254))
    return img

