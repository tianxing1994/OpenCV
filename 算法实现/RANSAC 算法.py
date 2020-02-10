"""
RANdom SAmple Consensus 随机抽样一致

https://blog.csdn.net/robinhjwy/article/details/79174914
https://blog.csdn.net/vict_wang/article/details/81027730
https://blog.csdn.net/l297969586/article/details/52328884
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


trans = PolynomialFeatures(degree=2)
linear_clf = LinearRegression()


def ransac(data, n, k, t, d):
    """
    data:二维数据
    model:训练好的模型
    n:随机抽取样本数目
    k:迭代次数
    t:阈值
    d:测试集大于多少才认为其是好的模型
    """
    data_size = len(data)
    epoch = 0
    best_model = None
    besterr = np.inf
    best_inlier_idxs = None
    while epoch < k:
        maybe_idxs, test_idxs = shuffle_data(data_size, n)
        maybe_inliers = data[maybe_idxs, :]
        test_inlier = data[test_idxs, :]
        maybeModel = linear_clf.fit(trans.fit_transform(maybe_inliers[:, :-1]), maybe_inliers[:, -1])
        test_error, _ = get_error(maybeModel, test_inlier[:, :-1], test_inlier[:, -1])
        also_idxs = test_idxs[test_error < t]
        also_inliers = data[also_idxs, :]
        if len(also_inliers) > d:
            better_data = np.concatenate((maybe_inliers, also_inliers))
            betterModel = linear_clf.fit(trans.fit_transform(better_data[:, :-1]), better_data[:, -1])
            _, thisError = get_error(betterModel, better_data[:, :-1], better_data[:, -1])
            if thisError < besterr:
                best_model = betterModel
                besterr = thisError
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
        epoch += 1
    if best_model is None:
        raise ValueError("无法拟合出 model")
    else:
        return best_model, besterr, best_inlier_idxs


def shuffle_data(data_row,n):
    idxs = np.arange(data_row)
    np.random.shuffle(idxs)
    return idxs[:n], idxs[n:]


def get_error(model, test, y_true):
    y_predict = model.predict(test)
    error = np.sqrt((y_predict-y_true)**2)
    mean_error = np.mean(error)
    return error, mean_error

