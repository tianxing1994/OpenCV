"""
参考链接:
https://blog.csdn.net/weixin_36913190/article/details/80490738

在 KMeans.fit() 和 KMeans.predict() 时都有 sample_weight 样本权重.
在 fit() 时, 要计算聚类中心, 此时的权重应在 np.mean() 计算均值向量处.
但是 predict() 时呢 ? 计算样本到中心的距离, 这权重该起什么作用 ? 因此, 我没有加入 sample_weight.
"""
import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn import cluster


class KMeans(object):
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._tol = tol

        self._X = None

        self._centers1 = None
        self._centers2 = None
        self._labels = None

    @staticmethod
    def random_center(dataset, k):
        index = random.sample(range(dataset.shape[0]), k)
        return dataset[index]

    @staticmethod
    def calc_max_move_distance(centers1, centers2):
        """求出各聚类中心的移动距离, 返回移动距离最大的值."""
        max_move_distance = np.max(np.sum(np.square(centers1 - centers2, dtype=np.float64), axis=1))
        return max_move_distance

    def _find_centers(self, centers):
        """
        :param X: ndarray, 形状为 (m, n)
        :param centers: ndarray, 形状为 (k, n)
        :return:
        """
        # 形状为 (m, n, 1)
        dataset = np.expand_dims(self._X, axis=2)
        # 形状为 (1, n, k)
        centers = np.expand_dims(centers.T, axis=0)
        distances = np.sqrt(np.sum(np.square(dataset - centers), axis=1))
        result = np.argmin(distances, axis=1)
        return result

    def _calc_centers(self, classify):
        k = len(np.unique(classify))
        centers = list()
        for i in range(k):
            samples = self._X[classify == i]
            center = np.mean(samples, axis=0)
            centers.append(center)
        result = np.array(centers)
        return result

    def fit(self, X, y=None, sample_weight=None):
        self._X = X
        self._centers2 = self.random_center(self._X, self._n_clusters)
        for _ in range(self._max_iter):
            self._labels = self._find_centers(self._centers2)
            self._centers1 = self._centers2
            self._centers2 = self._calc_centers(self._labels)
            max_move_distance = self.calc_max_move_distance(self._centers1, self._centers2)
            if max_move_distance < self._tol:
                break

        self.cluster_centers_ = self._centers2
        self.labels_ = self._labels

    def predict(self, X, sample_weight=None):
        result = self._find_centers(self.cluster_centers_)
        return result


def demo1():
    iris = load_iris()
    data = iris.data
    # target = iris.target
    kmeans = KMeans(n_clusters=3, max_iter=300)
    kmeans.fit(data)
    result = kmeans.predict(data)

    print(kmeans.cluster_centers_)
    # print(kmeans.labels_)
    # print(result)
    return


def demo2():
    iris = load_iris()
    data = iris.data
    # target = iris.target
    kmeans = cluster.KMeans(n_clusters=3, max_iter=300, tol=1e-4)
    kmeans.fit(data)
    result = kmeans.predict(data)

    print(kmeans.cluster_centers_)
    # print(kmeans.labels_)
    # print(result)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
