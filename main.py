from io import BytesIO

import requests
import torch
import torchvision
from PIL import Image

# def get_image(imageurl):
#     # 从通过http获取图片
#     proxies = {"http": None, "https": None}
#     req = requests.get(imageurl, verify=False, proxies=proxies)
#     filename = imageurl.split('/')[-1]
#     image = Image.open(BytesIO(req.content))
#     # image.save(filename)
#     return image
#
# imageurl = 'https://yiyi-picture.oss-cn-hangzhou.aliyuncs.com/Typora/001_1.png'
# # 从通过http获取图片
# proxies = { "http": None, "https": None }
# req = requests.get(imageurl, verify=False, proxies=proxies)
# filename = imageurl.split('/')[-1]
# image = Image.open(BytesIO(req.content))
# image.show()
# image.save(filename)

model = torchvision.models.densenet201(pretrained=False)
print(model)
# model.fc = torch.nn.Linear(512, 10, bias=True)
# print(model.fc)
# model.fc = torch.nn.Linear(2048, 10, bias=True)
# print(model)



