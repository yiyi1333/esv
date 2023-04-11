import torch
import torchvision
import os
from PIL import Image, ImageOps
from torchvision import transforms

from net import VGG_16

if torch.cuda.is_available():
    #打印GPU信息
    print(torch.cuda.get_device_name(0))
gpu_available = torch.cuda.is_available()

# 准备模型VGG
model = torchvision.models.vgg16(pretrained=False)
model.classifier[6] = torch.nn.Linear(4096, 5, bias=True)
# 加载参数
model.load_state_dict(torch.load('../model/result/vgg16_5.pth'))
if gpu_available:
    model = model.cuda()

# 推理模式
model.eval()
image = Image.open('../src/003_29.png')
# Image，以原图为中心，填充为正方形，使用白色填充
edge = max(image.size[0], image.size[1])
image = ImageOps.pad(image, (edge, edge), color=(255, 255, 255))
# resize 为224 * 224
image = image.resize((224, 224))
tran = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.GaussianBlur(3, (0.1, 2.0))
])
# totensor
image = tran(image)
# unsqueeze
image = image.unsqueeze(0)
# 进行推理
if gpu_available:
    image = image.cuda()
result = model(image)

print(result)
# 打印结果中最大的值
print(torch.max(result, 1))
