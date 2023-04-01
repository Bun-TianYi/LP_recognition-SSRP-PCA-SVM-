from Tool.utils import *
import cv2 as cv
import argparse
from joblib import load
import time


'''
    请注意，本程序所用所有坐标，均遵循左上角为坐标起点，右下角为坐标终点
    本程序所有矩阵坐标，左边永远指代行数，右边永远指代列数
    本程序所有矩形框的坐标（x，y，w，h）分别指代列起点，行起点，列增量，行增量
'''

start = time.time()
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


parser = argparse.ArgumentParser(description="model_output")
parser.add_argument('--dataset_path', type=str, default='C:/Users/user/Desktop/Digital_image_job/CCPD2019',
                    help='directory of data')
parser.add_argument('--n_compose', type=int, default=8, help='设定提取出的主成分数量')
parser.add_argument('--save_path', type=str, default='./division_data', help='分割数据的保存目录')

args = parser.parse_args()  # 生成参数，后续可在arg参数中找到这些设定好的参数的值与对应解释
img_path = args.dataset_path + '/ccpd_db'
datasets = Dataloader(img_path, mylen=True, data_len=[0, 100])
clf = load('SVMmodel.joblib')  # 加载svm模型

# 读取照片，这里要根据具体情况设置路径
image, image_name = datasets.__getitem__(8)

# ss区域框推荐模块
print('开始对图片进行区域推荐')
image_cv = (image*255).astype('uint8')
image_cv2 = (image*255).astype('uint8')

# 等比例缩放图片后交由cv库进行区域推荐
newHeight = 300
newWidth = int(image_cv.shape[1] * newHeight / image_cv.shape[0])
trans_h_rate = image_cv.shape[0] / newHeight  # 保存缩放率
trans_w_rate = image_cv.shape[1] / newWidth
image_cv = cv.resize(image_cv, (newWidth, newHeight))
ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image_cv)
ss.switchToSelectiveSearchFast()
rects = ss.process()  # 这里的存放预测出的区域框

rects[:, [0, 2]] = (rects[:, [0, 2]] * trans_h_rate).astype('int')  # 由于进行了缩放，需要把坐标缩放率乘回去
rects[:, [1, 3]] = (rects[:, [1, 3]] * trans_w_rate).astype('int')

print('区域推荐完毕，开始生成车牌区域')
# img_p = img_pca(image, args.n_compose, isfeature=False)
# img_p = img_p.transpose(1, 2, 0).astype('uint8')
# cv.imshow('41', img_p)
# cv.waitKey(0)

pre_proba = 0
tag = 0
for k, (box) in enumerate(rects):
    x, y, w, h = box
    # 将图片的通道与行列值换个位置，这是由于cv将numpy的行视为列导致的
    img = image[y: y + h + 1, x: x + w + 1, :]
    img = zero_resize(img, 128, 256)

    # 对图片进行主成分分析
    img_PCA = img_pca(img, args.n_compose, isfeature=True)

    # 获得主成分特征后将其拉成一维向量，取实部，方便svm预测
    img_feature = np.real(img_PCA.flatten()).reshape(1, -1)

    pre_label = int(clf.predict(img_feature))

    if pre_label == 1:
        temp = clf.predict_proba(img_feature)
        temp = temp[0, 1]
        if pre_proba < temp:
            pre_proba = temp
            tag = k     # 记录最大置信度的框的索引
    # cv.imshow('ll', img)
    # cv.waitKey(0)
    # p = 0

# 输出最大预测概率的预测框，并保存为相应文件
x, y, w, h = rects[tag]
img_cv = image_cv2.copy()
cv.rectangle(img_cv, (x, y), (x + w + 1, y + h + 1), (0, 0, 255), 2)
cv.imwrite('out.jpg', img_cv)
end = time.time()
print('运行时间为：' + str(abs(start - end)) + 's')