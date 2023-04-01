from Tool.utils import *
import numpy as np
import cv2 as cv


def img_pca(image, n_compose, isfeature=False):
    # 分别将图片送入PCA中进行一次计算，得到输出图之后再合并起来，并将数值还原
    img = np.array([my_PCA(image[:, :, 0], n_compose, isfeature),
                    my_PCA(image[:, :, 1], n_compose, isfeature),
                    my_PCA(image[:, :, 2], n_compose, isfeature)]) * 255
    return img.transpose(1, 2, 0)


img = cv.imread('test_image2.jpg')
img2 = zero_resize(img, 128, 256).astype('float32')/255
img3 = img_pca(img2, 6, isfeature=False)
cv.imshow('ll', img3.astype('uint8'))
cv.waitKey(0)
