# coding=utf8
# 添加环境变量.
import os
import sys
p = os.getcwd()
sys.path.append(p)

"""
1. 生成的样本的 shape 为: (2898, 36).
2. SVM 的输入数据 data 必须为 float
3. 最后的精度为: 0.9986197377501725
"""
import cv2 as cv
import numpy as np

from cv_hog import get_hog_dataset
from cv_svm import get_empty_svm
from load_data.screws_dataset import get_image_dataset


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def score(data, target):
    _, target_ = svm.predict(data)
    score = np.sum(target_ == target) / target.shape[0]
    return score


data, target_ = get_hog_dataset(*get_image_dataset())
target = np.expand_dims(target_, axis=1)

print("data 训练数据的形状: ", data.shape)
print("target 训练数据的形状: ", target.shape)

print("训练数据标记为 0 的样本的数量: ", np.sum(target_ == 0))
print("训练数据标记为 1 的样本的数量: ", np.sum(target_ == 1))
print("训练数据标记为 2 的样本的数量: ", np.sum(target_ == 2))

svm = get_empty_svm()
svm.train(data, cv.ml.ROW_SAMPLE, target)

precision = score(data, target)
print("svm 训练结果的精度: ", precision)
