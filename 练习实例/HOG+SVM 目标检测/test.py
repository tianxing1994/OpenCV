# coding=utf8
import cv2 as cv
import numpy as np

from config import win_size, win_stride
from config import hog
# from svm_classify import get_empty_svm, get_svm_detector
from load_data.hog_dataset import get_hog_dataset


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def get_svm_detector(svm):
    """
    从训练好的 SVM 分类器中取出支持向量和 rho 参数.
    导出可以用于 cv2.HOGDescriptor() 的 SVM 检测器, 实质上是训练好的 SVM 的支持向量和 rho 参数组成的列表.
    :param svm: 训练好的 SVM 分类器.
    :return: SVM 的支持向量和 rho 参数组成的列表, 可用作 cv2.HOGDescriptor() 的 SVM 检测器.
    """
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)


def get_empty_svm():
    svm = cv.ml.SVM_create()
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)    # for EPSILON_SVR, epsilon in loss function?
    svm.setC(0.01)    # From paper, soft classifier
    svm.setType(cv.ml.SVM_EPS_SVR)
    return svm


data, target = get_hog_dataset()
# data = np.array([[1, 1, 1], [0, 0, 0]])
# target = np.array([1, 0])
print(data.shape)
print(target.shape)
print(data.dtype)
print(target.dtype)
print(np.max(target))
print(np.min(target))
svm = get_empty_svm()
svm.train(data, cv.ml.ROW_SAMPLE, target)
svm_detector = get_svm_detector(svm)


hog.setSVMDetector(svm_detector)

image_path = "dataset/image/luosi.jpg"
image = cv.imread(image_path)
gray = cv.cvtColor(image_path, cv.COLOR_BGR2GRAY)
foundLocations, foundWeights = hog.detectMultiScale(gray, winStride=win_stride, scale=1.05)

print("foundLocations: ", foundLocations)
print("foundWeights: ", foundWeights)
for (x, y, w, h), weight in zip(foundLocations, foundWeights):
    if weight > 0.1:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

show_image(image)
