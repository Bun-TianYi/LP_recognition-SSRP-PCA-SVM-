# 定义一个一般的数据集处理类
import numpy as np
from torch.utils import data
import cv2 as cv
import xml.etree.ElementTree as et
import os
from torchvision.transforms import transforms
import glob


def iou_calculate(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]
    boxA[2] = boxA[0] + boxA[2]
    boxA[3] = boxA[1] + boxA[3]
    boxB[2] = boxB[0] + boxB[2]
    boxB[3] = boxB[1] + boxB[3]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def tensor2imag(img):
    # 这个函数将图像从tensor转换为image
    img = img.squeeze(0).numpy()
    img = img * 255  # 传入的矩阵一定为0-1区间的值，需要扩张到0-255
    mat = img.astype('uint8')  # 将矩阵转为uint8类型
    mat = mat.transpose(1, 2, 0)  # 调转矩阵维度
    return mat


def np_load_img(filename, resized_w, resized_h, resized=False):
    img = cv.imread(filename)
    if resized:
        img = cv.resize(img, (resized_w, resized_h))
    img = img.astype(dtype=np.float32)
    img = img / 255  # 将图片像素压缩到（0，1）区间
    return img


# test = "./VOCdevkit/VOC2012/Annotations/2012_004328.xml"
# test.replace('\\', '/')
# o = np_load_label(test)
class Dataloader(data.Dataset):
    def __init__(self, folder_path, resized=False, resize_w=224, resize_h=224, mylen=False, data_len=[0, 10]):
        self.folder_path = folder_path
        if mylen:
            self.data_len = data_len
        else:
            self.data_len = [0, -2]
        self.resized = resized
        self.resized_w = resize_w
        self.resized_h = resize_h
        self.samples = self.get_all_samples()

    def get_all_samples(self):
        img_list = []
        path = self.folder_path + '/*.jpg'
        img_names = glob.glob(path)
        # 这一步直接读取指定路径文件夹底下的所有以jpg为结尾的图片
        for k, i in enumerate(sorted(img_names)):
            if (self.data_len[1] - self.data_len[0] + 1) == 0:
                break
            elif k < self.data_len[0]:
                continue
            else:
                self.data_len[0] = self.data_len[0] + 1
            i = i.replace('\\', '/')  # 将文件路径中的“\\”替换为”/“，方便接下来的操作
            img_list.append(i.split('/')[-1])
        return img_list

    def __getitem__(self, item):
        img_path = self.folder_path + '/' + self.samples[item]
        img = np_load_img(img_path, self.resized_w, self.resized_h, resized=self.resized)
        # dimg = self.img2tensor(img)
        return img, self.samples[item]

    def __len__(self):
        return len(self.samples)

    def img2tensor(self, img):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        return transform(img)


def my_PCA(x, n_compose, isfeature=False):
    n_samples, n_features = x.shape
    mean = np.array([np.mean(x[:, i]) for i in range(n_features)])
    # 这里相当于是把x矩阵按列求平均值，就是类比于将每个特征进行均值归一化处理
    norm_x = (x - mean)
    xTx = np.dot(norm_x.T, norm_x)
    eig_value, eig_vector = np.linalg.eig(xTx)

    # 这个写法很独特，等于是将特征值与一个特征向量绑定在一起形成一个元组，
    # 而sort函数默认以一个元组中的第一个元素作为排序依据进行排序
    # eig_pair = [(np.abs(eig_value[i]), eig_vector[:, i]) for i in range(n_features)]
    # eig_pair.sort(reverse=True)
    # feature = np.array([ele[1] for ele in eig_pair[:n_compose]])
    vector_index = np.argsort(-eig_value)
    feature = eig_vector[:, vector_index[0: n_compose]]
    feature = feature.T
    rec_data = np.dot(norm_x, feature.T)
    img_pca = np.dot(rec_data, feature) + mean
    if isfeature:
        return rec_data
    else:
        return img_pca


def zero_resize(mat, new_width, new_height):
    # 这个函数仅接受传进来一个二维矩阵，若是三维矩阵请使用拼接技术完成相关工作
    if len(mat.shape) == 3:
        img = np.array([zero_resize(mat[:, :, 0], new_width, new_height),
                  zero_resize(mat[:, :, 1], new_width, new_height),
                  zero_resize(mat[:, :, 2], new_width, new_height)])
        return img.transpose(1, 2, 0)
    else:
        width, height = mat.shape
        new_mat = np.zeros((new_width, new_height))  # 生成一个新的矩阵，等下就返回这个矩阵
        if width / height > new_width / new_height:
            #   这是将图像尺寸等比例映射到我们需要它映射到的空间，剩下的地方补0即可
            temp_width = new_width
            temp_height = int(height * temp_width / width)
        else:
            temp_height = new_height
            temp_width = int(width * temp_height / height)

        mat = cv.resize(mat, (temp_height, temp_width))
        new_mat[0: temp_width, 0:temp_height] = mat

        return new_mat
