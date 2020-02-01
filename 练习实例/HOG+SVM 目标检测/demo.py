# coding=utf8
"""
SVM 的输入训练数据 data 必须为 float
"""

# 添加环境变量.
import os
import sys
p = os.getcwd()
sys.path.append(p)


import cv2 as cv
import numpy as np

from cv_hog import get_hog_dataset, hog, win_stride, custom_hog_detect
from cv_svm import get_empty_svm, get_svm_detector
from load_data.screws_dataset import get_image_dataset


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def score(svm, data, target):
    _, target_ = svm.predict(data)
    score = np.sum(target_ == target) / target.shape[0]
    return score


def hog_detectMultiScale(hog, image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    foundLocations, foundWeights = hog.detectMultiScale(gray, winStride=win_stride, scale=1.05)

    print("foundLocations: ", foundLocations)
    print("foundWeights: ", foundWeights)
    for (x, y, w, h), weight in zip(foundLocations, foundWeights):
        if weight > 0.5:
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image


def hog_detect(hog, image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    foundLocations, foundWeights = hog.detect(gray, winStride=win_stride)

    print("foundLocations: ", foundLocations)
    print("foundWeights: ", foundWeights)
    for (x, y), weight in zip(foundLocations, foundWeights):
        if weight > 2.7:
            cv.circle(image, center=(x, y), radius=2, color=(0, 0, 255), thickness=2)
    return image


def demo1():
    """SVM 分类演示. """
    data, target_ = get_hog_dataset(*get_image_dataset())
    target = np.expand_dims(target_, axis=1)

    print("data 训练数据的形状: ", data.shape)
    print("target 训练数据的形状: ", target.shape)
    print("训练数据标记为 0 的样本的数量: ", np.sum(target_ == 0))
    print("训练数据标记为 1 的样本的数量: ", np.sum(target_ == 1))
    print("训练数据标记为 2 的样本的数量: ", np.sum(target_ == 2))

    svm = get_empty_svm()
    svm.train(data, cv.ml.ROW_SAMPLE, target)

    precision = score(svm, data, target)
    print("svm 训练结果的精度: ", precision)
    return


def demo2():
    """
    没跑通.
    hog.detect 好像只接受:
    1. Linear 核的 SVM.
    2. 二分类的 SVM.
    直接调用的效果很差.
    """
    data, target_ = get_hog_dataset(*get_image_dataset(cls_list=(1,)))
    target = np.expand_dims(target_, axis=1)
    svm = get_empty_svm()
    svm.train(data, cv.ml.ROW_SAMPLE, target)

    svm_detector = get_svm_detector(svm)
    # HOGDescriptor::SetSVMDetector() 有些限制: linear kernel only, imgsize == winsize
    hog.setSVMDetector(svm_detector)

    image_path = "dataset/image/luosi.jpg"
    image = cv.imread(image_path)

    image = hog_detect(hog, image)
    show_image(image)
    return


def demo3():
    """OpenCV 中 hog.detect 调用没有成功, 自己实现了一个检测方法"""
    data, target_ = get_hog_dataset(*get_image_dataset(cls_list=(1, 2)))
    target = np.expand_dims(target_, axis=1)
    svm = get_empty_svm()
    svm.train(data, cv.ml.ROW_SAMPLE, target)

    precision = score(svm, data, target)
    print("svm 训练结果的精度: ", precision)

    image_path = "dataset/image/luosi.jpg"
    image = cv.imread(image_path)

    # 对原图像进行细小的缩放以查看模型的健壮性.
    image = cv.resize(src=image, dsize=(0, 0), fx=1.2, fy=1.2)

    bounding_box_1, bounding_box_2 = custom_hog_detect(image=image, hog=hog, svm=svm)
    for bounding_box in bounding_box_1:
        x1, y1, x2, y2 = bounding_box
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for bounding_box in bounding_box_2:
        x1, y1, x2, y2 = bounding_box
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    show_image(image)
    return


if __name__ == '__main__':
    # demo1()
    demo3()
