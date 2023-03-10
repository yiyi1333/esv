import os

from PIL import Image, ImageOps

forgery_path = 'images/Sig2011/Forgery'
genuine_path = 'images/Sig2011/Genuine'

forgery_list = os.listdir(forgery_path)
genuine_list = os.listdir(genuine_path)

# 遍历文件夹下的所有文件
# forgery_map = {}
# genuine_map = {}
# for forgery in forgery_list:
#     num = forgery.split('_')[0][-3:]
#     if(num not in forgery_map):
#         forgery_map[num] = 1
#     else:
#         forgery_map[num] += 1
#
# for genuine in genuine_list:
#     num = genuine.split('_')[0]
#     if(num not in genuine_map):
#         genuine_map[num] = 1
#     else:
#         genuine_map[num] += 1
#
# # 按关键词排序
# forgery_map = sorted(forgery_map.items(), key=lambda x: x[0])
# genuine_map = sorted(genuine_map.items(), key=lambda x: x[0])
# print(forgery_map)
# print(genuine_map)

# 遍历文件夹下的所有文件，计算图片的长和宽的最大值
# max_width = 0
# max_height = 0
# for forgery in forgery_list:
#     if forgery.endswith(".jpg") or forgery.endswith(".png"):
#         # 打开文件PIL打开图片
#         image = Image.open(os.path.join(forgery_path, forgery))
#         width, height = image.size
#         if(width > max_width):
#             max_width = width
#         if(height > max_height):
#             max_height = height
#
# for genuine in genuine_list:
#     if genuine.endswith(".jpg") or genuine.endswith(".png"):
#         image = Image.open(os.path.join(genuine_path, genuine))
#         width, height = image.size
#         if(width > max_width):
#             max_width = width
#         if(height > max_height):
#             max_height = height
#
# print(max_width, max_height)
