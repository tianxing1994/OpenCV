"""
参考链接:
https://blog.csdn.net/qq_34784753/article/details/61917999
"""
from collections import Counter
import numpy as np
from sklearn.datasets import load_iris
from sklearn import neighbors


class KNeighborsClassifier(object):
    def __init__(self, n_neighbors=5):
        self._n_neighbors = n_neighbors

        self._X_train = None
        self._y_train = None

    def fit(self, X, y):
        self._X_train = X
        self._y_train = y

    def predict(self, X):
        result = list()
        for X_i in X:
            k_X_train, k_y_train = self._get_neighbors(self._X_train, self._y_train, X_i, self._n_neighbors)
            classify = self._get_response(k_y_train)
            result.append(classify)
        return np.array(result)

    def score(self, X, y, sample_weight=None):
        """
        计算平均准确率
        :param X: 测试集样本, ndarray, 形状为 (n_samples, n_features)
        :param y: 测试集样本的真实类别, ndarray, 形状为 (n_samples,)
        :param sample_weight: 样本权重, ndarray, 形状为 (n_samples,)
        :return:
        """
        y_predict = self.predict(X)
        temp_result = np.array(y_predict == y, dtype=np.int)
        if sample_weight is None:
            result = np.sum(temp_result) / len(temp_result)
        else:
            result = np.sum(temp_result * sample_weight) / np.sum(sample_weight)
        return round(result, 4)

    @staticmethod
    def _get_neighbors(X_train, y_train, test_sample, k):
        """
        从训练样本 X_train 中选出 k 个与 test_sample 距离最小的训练样本. 并返回对应的 k 个 (k_X_train, k_y_train)
        :param X_train: 已知分类的训练集. 形状为 (m, n)
        :param test_sample: 当前样本. 形状为 (n, )
        :param k: int, 查找 k 个与 test_sample 最邻近的训练样本.
        :return:
        """
        distances = np.sqrt(np.sum((X_train - np.expand_dims(test_sample, axis=0)) ** 2, axis=1))
        argsort = np.argsort(distances)
        index = argsort[:k]
        return X_train[index], y_train[index]

    @staticmethod
    def _get_response(k_y_train):
        """
        计算 k_y_train 中各类别样本的数量. 返回数量最多的类别的值.
        :param k_y_train: 训练集样本
        :return:
        """
        counter = Counter(k_y_train)
        counter_l = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        result = counter_l[0][0]
        return result


def demo1():
    """准确率 0.9867"""
    iris = load_iris()
    data = iris.data
    target = iris.target

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(data, target)
    # result = knn.predict(data)
    # print(result)
    result = knn.score(data, target)
    print(result)
    return


def demo2():
    """sklearn 演示, 准确率 0.98"""
    iris = load_iris()
    data = iris.data
    target = iris.target
    knn = neighbors.KNeighborsClassifier(n_neighbors=10)
    knn.fit(data, target)
    result = knn.score(data, target)
    print(result)
    return


if __name__ == '__main__':
    demo1()
