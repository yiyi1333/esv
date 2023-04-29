import requests
from PIL import Image, ImageOps
from io import BytesIO
def get_image(imageurl):
    # 从通过http获取图片
    proxies = {"http": None, "https": None}
    req = requests.get(imageurl, verify=False, proxies=proxies)
    # filename = '../temp/' +  imageurl.split('/')[-1]
    image = Image.open(BytesIO(req.content))
    # image.save(filename)
    return image