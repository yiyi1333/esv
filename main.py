from io import BytesIO

import requests
import torch
import torchvision
from PIL import Image

def get_file(file_url):
    # 从通过http获取图片
    proxies = {"http": None, "https": None}
    req = requests.get(file_url, verify=False, proxies=proxies)
    filename = '../temp/' +  file_url.split('/')[-1]
    # 保存文件
    with open(filename, 'wb') as f:
        f.write(req.content)
    return filename


get_file('https://yiyi-picture.oss-cn-hangzhou.aliyuncs.com/image/2023-05-20-15-36-24.679117400-new-test-X.npy')



