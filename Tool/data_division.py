import json
import random

from Tool.utils import *
import argparse
import pandas as pd


'''
    请注意，本程序所用所有坐标，均遵循左上角为坐标起点，右下角为坐标终点
    本程序所有矩阵坐标，左边永远指代行数，右边永远指代列数
    本程序所有矩形框的坐标（x，y，w，h）分别指代列起点，行起点，列增量，行增量
'''


def get_box_point(img_name):
    # 这个函数获取图像名字中自带的图像尺寸与车牌位置信息
    img_name = img_name.split('.')[0]
    # 根据图像名分割标注
    _, _, box, points, label, brightness, blurriness = img_name.split('-')
    # --- 边界框信息
    box = box.split('_')
    box = [list(map(int, i.split('&'))) for i in box]
    # --- 关键点信息
    points = points.split('_')
    points = [list(map(int, i.split('&'))) for i in points]
    # 将关键点的顺序变为从左上顺时针开始
    # 请注意这里的坐标请以x为行，y为列来看待
    points = points[-2:] + points[:2]

    return box, points


def img_pca(image, n_compose, isfeature=False):
    # 分别将图片送入PCA中进行一次计算，得到输出图之后再合并起来，并将数值还原
    img = np.array([my_PCA(image[:, :, 0], n_compose, isfeature),
                     my_PCA(image[:, :, 1], n_compose, isfeature),
                     my_PCA(image[:, :, 2], n_compose, isfeature)]) * 255
    return img


parser = argparse.ArgumentParser(description="main")# 添加一个名叫gpus的parser参数，‘narg=+’ 指定该参数可以被传入多个后形成一个列表，但不接受不传如该参数
parser.add_argument('--resize_h', type=int, default=224, help='height of resize images')
parser.add_argument('--resize_w', type=int, default=224, help='width of resize images')
parser.add_argument('--dataset_path', type=str, default='C:/Users/user/Desktop/Digital_image_job/CCPD2019',
                    help='directory of data')
parser.add_argument('--n_compose', type=int, default=8, help='设定提取出的主成分数量')
parser.add_argument('--save_path', type=str, default='./division_data', help='分割数据的保存目录')
parser.add_argument('--iou', type=float, default=0.65, help='iou评判阈值')


args = parser.parse_args()
img_path = args.dataset_path + '/ccpd_base'
args = parser.parse_args()  # 生成参数，后续可在arg参数中找到这些设定好的参数的值与对应解释
datasets = Dataloader(img_path, mylen=True, data_len=[0, 20000])

img_labels = []
img_features = []
with open('../rect_dict.json', 'r') as f:
    box_bnd = json.load(f)

for item in range(1500):
    # 读取照片，这里要根据具体情况设置路径
    image, image_name = datasets.__getitem__(item)
    _, point = get_box_point(image_name)
    point = np.array(point)
    min_x = min(point[:, 0])
    min_y = min(point[:, 1])
    max_x = max(point[:, 0])
    max_y = max(point[:, 1])
    label_point = [min_x, min_y, max_x - min_x, max_y - min_y]
    for box in box_bnd[str(item)]:
        x, y, w, h = box
        img_iou = iou_calculate(box, label_point)
        if img_iou <= args.iou:
            seed = random.random()
            if seed < 0.01:
                img_label = -1
            else:
                continue
        else:
            img_label = 1
        # 将图片的通道与行列值换个位置，这是由于cv将numpy的行视为列导致的
        img = image[y: y + h + 1, x: x + w + 1, :]
        img = zero_resize(img, 128, 256)
        img_PCA = img_pca(img, args.n_compose, isfeature=True)
        img_feature = np.real(img_PCA.flatten())
        img_labels.append(img_label)
        img_features.append(img_feature)


    print('第' + str(item) + '图片已处理完毕')
    # cv.imshow('ll', img)
    # cv.waitKey(0)
    # p = 0

label_len = len(img_labels)
img_features = np.array(img_features)
img_labels = np.array([img_labels]).T
# 将百分之八十的数据划分为训练集，百分之二十的数据划分为测试集

shuffle_index = list(np.arange(label_len))
random.shuffle(shuffle_index)
img_labels = img_labels[shuffle_index]
img_features = img_features[shuffle_index]
train_features = img_features[0:int(label_len*0.8)]     # 选取数组的前80%个元素作为训练集
test_features = img_features[int(label_len*0.8):]       # 选取前20%作为测试集
train_data = np.concatenate((train_features, img_labels[0:int(label_len*0.8)]), axis=1)
test_data = np.concatenate((test_features, img_labels[int(label_len*0.8):]), axis=1)
train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)
train_data.to_pickle('train_data.pkl')
test_data.to_pickle('test_data.pkl')


