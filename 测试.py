import argparse
import json
import os
import random
import cv2 as cv
import numpy as np

from Tool.utils import *

parser = argparse.ArgumentParser(description="Data preprocessing")
# 添加一个名叫gpus的parser参数，‘narg=+’ 指定该参数可以被传入多个后形成一个列表，但不接受不传如该参数
parser.add_argument('--resize_h', type=int, default=224, help='height of resize images')
parser.add_argument('--resize_w', type=int, default=224, help='width of resize images')
parser.add_argument('--dataset_path', type=str, default='C:/Users/user/Desktop/Digital_image_job/CCPD2019',
                    help='directory of data')
parser.add_argument('--save_path', type=str, default='./division_data', help='分割数据的保存目录')

args = parser.parse_args()  # 生成参数，后续可在arg参数中找到这些设定好的参数的值与对应解释

# # 开始设定gpu相关参数
# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"  # 按照PCI_BUS_ID顺序从0开始排列
# if args.gpus is None:
#     gpus = "0"  # 如果未指定GPU，那么就指定第0块gpu
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpus
# else:
#     gpus = ""
#     for i in range(len(args.gpus)):
#         gpus = gpus + args.gpus + ","
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]
# # 这段代码设定如果gpu只有一个，那就直接拿来用，如果有多个，那么就形成一个gpu列表进行调用
#
# torch.backends.cudnn.enabled = True  # 确保cudann起作用

# 设定图像与标签的数据集路径
img_path = args.dataset_path + '/ccpd_base'
# label_path = args.dataset_path + '/Annotations'


# 为计算加速
cv.setUseOptimized(True)
cv.setNumThreads(32)

rect_dict = {}
# 这里仅选取前2000张图片作为训练集（电脑撑不住啊）

image_cv = cv.imread('test1.jpg')
image = image_cv.copy()
# 重置图片大小，高设置为 300，保持高、宽比例

newHeight = 300
newWidth = int(image_cv.shape[1] * newHeight / image_cv.shape[0])
trans_h_rate = image_cv.shape[0] / newHeight  # 保存缩放率
trans_w_rate = image_cv.shape[1] / newWidth
image_cv = cv.resize(image_cv, (newWidth, newHeight))

# 创建 Selective Search Segmentation 对象
ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()

# 添加待处理的图片
ss.setBaseImage(image_cv)

# 可以选择快速但是低 recall 的方式
# 这里的 recall 指的是选择出来的 region 是否包含了所有应该包含的区域。recall 越高越好
ss.switchToSelectiveSearchFast()

# 也可以选择慢速但是高 recall 的方式
# ss.switchToSelectiveSearchQuality()
# 进行 region 划分，输出得到的 region 数目
rects = ss.process()  # 这里的区域是已经生成完毕的，不可能更加多了
rects[:, [0, 2]] = (rects[:, [0, 2]] * trans_h_rate).astype('int')  # 由于进行了缩放，需要把坐标缩放率乘回去
rects[:, [1, 3]] = (rects[:, [1, 3]] * trans_w_rate).astype('int')

image2 = image.copy()
for k, (rect) in enumerate(rects):
    seed = random.random()
    x, y, w, h = rect
    cv.rectangle(image, (x, y), (x+w, y+h), color=[0, 255, 85], thickness=1)
    if seed < 0.01:
        img_temp = image2[y:y + h + 1, x:x + w + 1, :]
        cv.imwrite('split' + str(k) + '.jpg', img_temp)
cv.imwrite('output.jpg', image)