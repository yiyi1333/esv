from io import BytesIO

import requests
import torch
import torchvision
from PIL import Image, ImageOps
from flask import Flask, request, jsonify
import json

from torchvision import transforms

app = Flask(__name__)

def get_image(imageurl):
    # 从通过http获取图片
    proxies = {"http": None, "https": None}
    req = requests.get(imageurl, verify=False, proxies=proxies)
    filename = '../temp/' +  imageurl.split('/')[-1]
    image = Image.open(BytesIO(req.content))
    image.save(filename)
    return filename

@app.route("/vgg16", methods=['POST'])
def vgg16():
    # 调用vgg16模型进行推理
    data = request.get_json()
    modelId = data['modelId']
    imageurl = data['imageUrl']

    print('imageUrl' + imageurl)
    # 从通过url获取图片
    imagepath = get_image(imageurl)

    if torch.cuda.is_available():
        # 打印GPU信息
        print(torch.cuda.get_device_name(0))
    gpu_available = torch.cuda.is_available()

    # 准备模型VGG
    model = torchvision.models.vgg16(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, 10, bias=True)
    # 加载参数
    model.load_state_dict(torch.load('../model/result/vgg16_5_2_1.pth'))
    if gpu_available:
        model = model.cuda()

    # 推理模式
    model.eval()

    image = pre_progress(imagepath)
    # 进行推理
    if gpu_available:
        image = image.cuda()
    result = model(image)
    # 将result中的结果重新计算成和为1的置信度
    result = torch.softmax(result, dim=1)
    # 将result中的结果保留两位小数
    # result = torch.round(result * 100) / 100
    # print(result)
    # 将结果转换成json格式
    result = result.tolist()
    result = json.dumps(result)
    print(result)
    # 返回推理结果
    return result

def pre_progress(img_path):
    image = Image.open(img_path)
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

if __name__ == '__main__':
    app.run()