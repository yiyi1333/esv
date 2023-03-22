import os
import random

from PIL import Image, ImageOps

def enhancement(img):
    # 将图片顺时针或逆时针随机旋转10度以内
    img = img.rotate(random.randint(-10, 10))
    # 将图片放大或缩小到原来的0.9~1.1倍
    img = img.resize((int(img.size[0] * random.uniform(0.9, 1.1)), int(img.size[1] * random.uniform(0.9, 1.1))))
    # 将图片随机平移0~20像素
    img = ImageOps.expand(img, border=random.randint(0, 20), fill=(254, 254, 254))
    return img


enhance_num = 10

# 获取Sig2011的目录
images_dir_genuine = '../images/Sig2011/Genuine'
images_dir_forgery = '../images/Sig2011/Forgery'

# 目标目录
target_dir_genuine = '../images/Sig2011_enhancement/Genuine'
target_dir_forgery = '../images/Sig2011_enhancement/Forgery'

# 读取目录下的所有图片
genuine_source_list = os.listdir(images_dir_genuine)
forgery_source_list = os.listdir(images_dir_forgery)

genuine_data_map = {}
for file in genuine_source_list:
    # print(images_dir_genuine + '/' + file)
    num = file.split('_')[0]
    image = Image.open(images_dir_genuine + '/' + file)
    if (num not in genuine_data_map):
        genuine_data_map[num] = 1
    else:
        genuine_data_map[num] += 1
    index = genuine_data_map[num]
    # 将图片image存储到目标目录
    image.save(target_dir_genuine + '/' + num + '_' + str(index) + '.png')
    # 对图片进行增强
    for i in range(enhance_num):
        image_new = enhancement(image)
        genuine_data_map[num] += 1
        index += 1
        image.save(target_dir_genuine + '/' + num + '_' + str(index) + '.png')

forgerty_data_map = {}
for file in forgery_source_list:
    num = file.split('_')[0]
    image = Image.open(images_dir_forgery + '/' + file)
    if(num not in forgerty_data_map):
        forgerty_data_map[num] = 1
    else:
        forgerty_data_map[num] += 1
    index = forgerty_data_map[num]
    # 将图片image存储到目标目录
    image.save(target_dir_forgery + '/' + num + '_' + str(index) + '.png')
    # 对图片进行增强
    for i in range(enhance_num):
        image_new = enhancement(image)
        forgerty_data_map[num] += 1
        index += 1
        image.save(target_dir_forgery + '/' + num + '_' + str(index) + '.png')


