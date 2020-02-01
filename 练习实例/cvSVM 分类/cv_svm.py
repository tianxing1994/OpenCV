# coding=utf8
import cv2 as cv
import numpy as np

from cv_hog import get_hog_dataset
from load_data.screws_dataset import get_image_dataset


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
    # https://blog.csdn.net/fengbingchun/article/details/78353140
    svm = cv.ml.SVM_create()
    svm.setGamma(1)
    svm.setCoef0(0.0)
    svm.setDegree(3)

    # 设置终止迭代条件.
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-2)
    svm.setTermCriteria(criteria)

    svm.setKernel(cv.ml.SVM_LINEAR)
    # svm.setKernel(cv.ml.SVM_RBF)

    svm.setNu(0.5)
    svm.setP(0.5)
    svm.setC(5)
    svm.setType(cv.ml.SVM_C_SVC)
    return svm
