import os
import random

from PIL import ImageOps, Image


def enhancement(img):
    # 将图片顺时针或逆时针随机旋转8度以内,使用白色填充
    img = img.rotate(random.randint(-4, 4), fillcolor=(255, 255, 255))
    # 将图片放大或缩小到原来的0.9~1.1倍
    img = img.resize((int(img.size[0] * random.uniform(0.7, 1.1)), int(img.size[1] * random.uniform(0.9, 1.1))))
    # 将图片随机平移-40~40像素
    img = ImageOps.expand(img, border=random.randint(-30, 30), fill=(255, 255, 255))
    return img


enhance_num = 8

# 设置目录
image_dir_genuine = '../images/Eszjut2023/Genuine'
image_dir_forgery = '../images/Eszjut2023/Forgery'

# 目标目录
target_dir_genuine = '../images/Eszjut2023_enhancement/Genuine'
target_dir_forgery = '../images/Eszjut2023_enhancement/Forgery'

# 判断目标目录是否存在，不存在则创建
if not os.path.exists(target_dir_genuine):
    os.makedirs(target_dir_genuine)
if not os.path.exists(target_dir_forgery):
    os.makedirs(target_dir_forgery)

# 读取目录下的所有图片
genuine_source_list = os.listdir(image_dir_genuine)
forgery_source_list = os.listdir(image_dir_forgery)

# 真实签名数据字典
genuine_data_map = {}

for file in genuine_source_list:
    num = file.split('_')[0]
    image = Image.open(image_dir_genuine + '/' + file)
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
        image_new.save(target_dir_genuine + '/' + num + '_' + str(index) + '.png')

print("enhancement progress 50% !")


# 伪造签名数据字典
forgery_data_map = {}
for file in forgery_source_list:
    num = file.split('_')[1]
    image = Image.open(image_dir_forgery + '/' + file)
    if (num not in forgery_data_map):
        forgery_data_map[num] = 1
    else:
        forgery_data_map[num] += 1
    index = forgery_data_map[num]
    # 将图片image存储到目标目录
    image.save(target_dir_forgery + '/' + num + '_' + str(index) + '.png')
    # 对图片进行增强
    for i in range(enhance_num):
        image_new = enhancement(image)
        forgery_data_map[num] += 1
        index += 1
        image_new.save(target_dir_forgery + '/' + num + '_' + str(index) + '.png')

print("enhancement progress 100% !")

