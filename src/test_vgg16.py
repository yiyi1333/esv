import torch
import torchvision
import os
from PIL import Image, ImageOps
from torchvision import transforms

from net import VGG_16


# 图像预处理
def pre_progress(image):
    # image = Image.open(img_path)
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
    return image


def inference(image, model_path, index):
    if torch.cuda.is_available():
        # 打印GPU信息
        print(torch.cuda.get_device_name(0))
    gpu_available = torch.cuda.is_available()

    # 准备模型VGG
    model = torchvision.models.vgg16(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, 10, bias=True)
    # 加载参数
    model.load_state_dict(torch.load(model_path))
    if gpu_available:
        model = model.cuda()

    # 推理模式
    model.eval()

    image = pre_progress(image)
    # 进行推理
    if gpu_available:
        image = image.cuda()
    result = model(image)
    print('模型推理的直接结果： ' , result)
    result = torch.softmax(result, dim=1)
    print('模型推理的softmax结果： ', result)

    recognition_rate = result[0][index * 2] + result[0][index * 2 + 1]
    real_rate = result[0][index * 2] / recognition_rate

    # 计算最大值的下标
    # index = torch.argmax(result, dim=1)
    # 如果index为偶数
    # if index % 2 == 0:
    #     # 计算数组中以index和index+1为下标的两个数的和
    #     recognition_rate = result[0][index] + result[0][index + 1]
    # else:
    #     # 计算数组中以index和index-1为下标的两个数的和
    #     recognition_rate = result[0][index] + result[0][index - 1]
    index = torch.argmax(result, dim=1).item()
    response = {}
    response['recognitionRate'] = recognition_rate.item()
    response['index'] = index
    response['realRate'] = real_rate.item()
    return response
