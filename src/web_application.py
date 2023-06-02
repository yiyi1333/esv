import os.path
from io import BytesIO
from test_vgg16 import inference

import requests
import torch
import torchvision
from PIL import Image, ImageOps
from flask import Flask, request, jsonify
import json

from torchvision import transforms

from src.util import get_image

app = Flask(__name__)

# imageSyn
@app.route('/imageSyn', methods=['POST'])
def imageSyn():
    req_data = request.get_json()
    type = req_data['type']
    # 先判断当前环境目录下的 /images/WebSyn/ 目录是否存在，不存在则创建
    if not os.path.exists('../images/WebSyn'):
        os.makedirs('../images/WebSyn')
        print(' 创建目录 WebSyn 成功')

    image_syn_list = req_data['imageSynList']
    for image_syn in image_syn_list:
        images_url = image_syn['imagesUrl']
        user_id = image_syn['userId']
        if not os.path.exists('../images/WebSyn/' + user_id + '/' + type):
            os.makedirs('../images/WebSyn/' + user_id + '/' + type)
            print(' 创建目录 ' + user_id + '/' + type + ' 成功')
        for image_url in images_url:
            if not os.path.exists('../images/WebSyn/' + user_id + '/' + type + '/' + image_url.split('-')[-1]):
                # 从通过url获取图片
                image = get_image(image_url)
                image.save('../images/WebSyn/' + user_id + '/' + type + '/' + image_url.split('-')[-1])
                print(' 保存图片 ' + image_url.split('-')[-1] + ' 成功')
    return req_data

@app.route('/verification', methods=['POST'])
def verification():
    req_data = request.get_json()
    image_url = req_data['imageUrl']
    network = req_data['network']
    model_name = req_data['modelName']
    index = req_data['index']
    print('index:' + str(index))

    model_path = '../model/WebModel/' + network + '_' + model_name + '.pth'
    print(model_path)
    if not os.path.exists(model_path):
        print('模型不存在')
        return '模型不存在'

    image = get_image(image_url)
    result = inference(image, model_path, index)
    print(result)
    return result


if __name__ == '__main__':
    app.run()