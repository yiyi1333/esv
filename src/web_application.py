import os.path
from io import BytesIO

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
        if not os.path.exists('../images/WebSyn/' + user_id):
            os.makedirs('../images/WebSyn/' + user_id)
            print(' 创建目录 ' + user_id + ' 成功')
        for image_url in images_url:
             # 从通过url获取图片
            image = get_image(image_url)
            if not os.path.exists('../images/WebSyn/' + user_id + '/' + image_url.split('-')[-1]):
                image.save('../images/WebSyn/' + user_id + '/' + image_url.split('-')[-1])
            print(' 保存图片 ' + image_url.split('-')[-1] + ' 成功')

    return req_data

if __name__ == '__main__':
    app.run()