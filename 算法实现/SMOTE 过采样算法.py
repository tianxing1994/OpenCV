"""
参考链接:
https://blog.csdn.net/qq_41686130/article/details/87103856
"""
from collections import Counter
import random

from sklearn.neighbors import NearestNeighbors
from imblearn import over_sampling
import numpy as np


class SMOTE(object):
    def __init__(self, k_neighbors=5):
        self._k_neighbors = k_neighbors

    @staticmethod
    def calc_lack_quantity(y):
        """计算每个类别需要增加多少个样本."""
        y_unique = np.unique(y)
        y_quantity = [len(y[y == y_]) for y_ in y_unique]
        max_quantity = np.max(y_quantity)
        y_lack_quantity = [max_quantity - len(y[y == y_]) for y_ in y_unique]
        return y_unique, y_quantity, y_lack_quantity

    @staticmethod
    def calc_allocation(y_q, y_l):
        a = int(y_l // y_q)
        b = int(y_l % y_q)
        nd = np.zeros(shape=(y_q,), dtype=np.int)
        nd[:] = a
        population = list(range(y_q))
        np.random.shuffle(population)
        index = population[: b]
        nd[index] += 1
        return nd

    @staticmethod
    def synthetic_samples(X, y, index, k_neighbors, n):
        """
        :param X: 类别都为 y 的样本. ndarray. 形状 (n_samples, n_features)
        :param y: 类别, 标量. 如 0
        :param index: 用于合成样本的原样本的索引
        :param k_neighbors: 原样本的 k 邻近样本的索引, ndarray, 形状为 (1, k)
        :param n: 要合成的样本的数量
        :return: X_result, y_result. 返回合成的样本, ndarray, 以及各样本的类别. y_result 中的值都为 y
        """
        synthetic_list = list()
        for i in range(n):
            neighbor = np.random.choice(np.squeeze(k_neighbors))
            diff = X[index] - X[neighbor]
            synthetic = X[i] + random.random() * diff
            synthetic_list.append(synthetic)
        X_result = np.array(synthetic_list)
        y_result = np.array([y] * len(synthetic_list))
        return X_result, y_result

    def synthetic_samples_y_(self, X_, y_, resolve_map_):
        """
        :param X_: 类别都为 y 的样本. ndarray. 形状 (n_samples, n_features)
        :param y_: 类别, 标量. 如 0
        :param resolve_map_: 指定 X_ 中每一个样本分别需要合成多少个新的样本. ndarray, 形状为 (n_samples,)
        :return: X_result, y_result, 返回合成的样本, ndarray, 以及各样本的对应的类别.
        """
        synthetic_list_X = list()
        synthetic_list_y = list()
        m, _ = X_.shape
        knn = NearestNeighbors(n_neighbors=self._k_neighbors)
        knn.fit(X_)
        for index in range(m):
            k_neighbors = knn.kneighbors(np.expand_dims(X_[index], axis=0), return_distance=False)
            number = resolve_map_[index]
            synthetic_X, synthetic_y = self.synthetic_samples(X_, y_, index, k_neighbors, number)
            synthetic_list_X.append(synthetic_X)
            synthetic_list_y.append(synthetic_y)
        X_result = np.concatenate(seq=synthetic_list_X, axis=0)
        y_result = np.concatenate(seq=synthetic_list_y, axis=0)
        return X_result, y_result

    def resolve_quantity(self, y, y_unique, y_quantity, y_lack_quantity):
        """将每个类别所缺少的样本数量分配到各样本."""
        resolve_map = np.zeros(shape=(np.sum(y_quantity)), dtype=np.int)
        for y_, y_q, y_l in zip(y_unique, y_quantity, y_lack_quantity):
            allocation = self.calc_allocation(y_q, y_l)
            resolve_map[y == y_] = allocation
        return resolve_map

    def fit_resample(self, X, y):
        synthetic_list_X = list()
        synthetic_list_y = list()
        y_unique, y_quantity, y_lack_quantity = self.calc_lack_quantity(y)
        resolve_map = self.resolve_quantity(y, y_unique, y_quantity, y_lack_quantity)
        for y_, y_l in zip(y_unique, y_lack_quantity):
            if y_l == 0:
                continue
            X_ = X[y == y_]
            resolve_map_ = resolve_map[y == y_]
            synthetic_X, synthetic_y = self.synthetic_samples_y_(X_, y_, resolve_map_)
            synthetic_list_X.append(synthetic_X)
            synthetic_list_y.append(synthetic_y)
        X_result = np.concatenate(seq=(X, *synthetic_list_X), axis=0)
        y_result = np.concatenate(seq=(y, *synthetic_list_y), axis=0)
        return X_result, y_result


def timer(func):
    import time

    def inner(*args, **kwargs):
        time1 = time.time()
        result = func(*args, **kwargs)
        time2 = time.time()
        print(f"time cost: {time2 - time1}")
        return result
    return inner


@timer
def demo1():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.8], n_informative=3,
                               n_redundant=1, flip_y=0, n_features=20, n_clusters_per_class=1,
                               n_samples=1000, random_state=10)
    print('Original dataset shape %s' % Counter(y))

    sm = over_sampling.SMOTE(random_state=42)
    X_result, y_result = sm.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y_result))
    return


@timer
def demo2():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.8], n_informative=3,
                               n_redundant=1, flip_y=0, n_features=20, n_clusters_per_class=1,
                               n_samples=1000, random_state=10)
    print('Original dataset shape %s' % Counter(y))

    sm = SMOTE(k_neighbors=5)
    X_result, y_result = sm.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y_result))
    return


if __name__ == '__main__':
    # demo1()
    demo2()
