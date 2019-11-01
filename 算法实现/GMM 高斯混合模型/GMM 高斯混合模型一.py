"""
参考链接:
https://blog.csdn.net/weixin_42137700/article/details/90762583

这个人的代码真的很棒. 和 sklearn GaussianMixture 的结果一样, 还做了 gif 动图.

我把代码封装成了根 sklearn GaussianMixture 类似的调用方法.


笔记心得:
1. 在根据样本求概率的时候, 如: X = [1, 0, 1, 0, 1, 1, 1, 0, 0, 1]
我们会得出: 为 1 的概率为 6/10, 为 0 的概率为 4/10.
这里我们可以看作是, 如 X 的样本发生了 n 次. (n 代表足够多次)
即: 6n/10n = 6/10. 如此, 可见, 在重复足够多次的情况下, 1 发生的概率是 6/10.

2. 如样本 X, 由 A 产生 X 的概率是 0.4, 由 B 产生 X 的概率是 0.6.
则可以看作是: 样本 X 发生了 n 次(足够多次).
则, 这其中由 A 产生的样本占 0.4*n 次, 由 B 产生的样本占 0.6*n 次.

3. 如样本 Y = [[1, 0, 1], [1, 1, 1], [0, 1, 0]]
假设其中每一组样本由 A, B 产生的概率是: [[0.7, 0.3], [0.9, 0.1], [0.3, 0.7]]

则可以看作: 样本 Y 发生了 n 次(足够多次).
其中 A 产生了 0.7*n 组 [1, 0, 1], 0.9*n 组 [1, 1, 1], 0.3*n 组 [0, 1, 0]
总计 A 产生了 (0.7*n*2 + 0.9*n*3 + 0.3*n*1) 个 1, (0.7*n*1 + 0.9*n*0 + 0.3*n*2) 个 0. (0.7*n+0.9*n+0.3*n)*3 个样本
其中 B 产生了 0.3*n 组 [1, 0, 1], 0.1*n 组 [1, 1, 1], 0.7*n 组 [0, 1, 0]
总计 B 产生了 (0.3*n*2 + 0.1*n*3 + 0.7*n*1) 个 1, (0.3*n*1 + 0.1*n*0 + 0.7*n*2) 个 0. (0.3*n+0.1*n+0.7*n)*3 个样本
在 n 个 Y 样本的情况下, 即: 当下的样本足够多, 可以用于推测 A, B 产生 1/0 的概率.
则由此样本推测: A 产生 1 的概率为 (0.7*n*2 + 0.9*n*3 + 0.3*n*1)/(0.7*n+0.9*n+0.3*n)*3 = 4.4/5.7 = 0.77, A 产生 0 的概率为 0.23
则由此样本推测: B 产生 1 的概率为 (0.3*n*2 + 0.1*n*3 + 0.7*n*1)/(0.3*n+0.1*n+0.7*n)*3 = 1.6/3.3 = 0.48, A 产生 0 的概率为 0.52

4. 已知 A, B 分别产生 1 的概率为 0.77, 0.48.
A, B 产生分别产生样本 Y 中每组样本的概率为:
[[0.77^2, 0.48^2], [0.77^3, 0.48^3], [0.77^1, 0.48^1]] = [0.59, 0.23], [0.46, 0.11], [0.77, 0.48]
因为知道 Y 中样本不是由 A 产生就是由 B 产生的, 所以, Y 中每组样本分别由 A, B 产生的可能性为:
[[0.59/(0.59+0.23), 0.23/(0.59+0.23)], [0.46/(0.46+0.11), 0.11/(0.46+0.11)], [0.77/(0.77+0.48), 0.48/(0.77+0.48)]] =
[[0.72, 0.28], [0.81, 0.19], [0.62, 0.38]]

得出, Y 中每组样本分别由 A, B 产生的概率为:
[[0.72, 0.28], [0.81, 0.19], [0.62, 0.38]]


5. 高斯混合模型, 其实是一种聚类算法.
假设我们已知一个样本集是从三个高斯分布中取出并组合到一起的, 我们希望通过这些样本推测出这三个高斯分布的具体参数(均值, 方差).
则这里所使用的 EM 算法则是求解的方法.
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class GMM(object):
    def __init__(self, n_components, max_iter):
        self.n_components = n_components
        self.max_iter = max_iter

        self._X = None
        self._m = None
        self._n = None

        self._weights = None
        self._means = None
        self._covariances = None

        self._expectation_array = None
        self._weights_array = None

    @property
    def weights_(self):
        if self._weights is not None:
            return self._weights
        else:
            raise AttributeError('没有 weights_ 属性')

    @property
    def means_(self):
        if self._means is not None:
            return self._means
        else:
            raise AttributeError('没有 _means 属性')

    @property
    def covariances_(self):
        if self._covariances is not None:
            return self._covariances
        else:
            raise AttributeError('没有 _covariances 属性')

    def _initialize_clusters(self):
        kmeans = KMeans(n_clusters=self.n_components).fit(self._X)

        weights = list()

        covariances = list()
        for i in range(self.n_components):
            sub_X = self._X[kmeans.labels_ == i]
            weight = sub_X.shape[0]
            weights.append(weight)
            sub_covariance = np.dot(sub_X.T, sub_X) / weight
            covariances.append(sub_covariance)

        self._weights = np.array(weights)
        self._means = kmeans.cluster_centers_
        self._covariances = np.array(covariances)
        return

    def _initialize_clusters2(self):
        kmeans = KMeans(n_clusters=self.n_components).fit(self._X)

        weights = [1.0 / self.n_components for _ in range(self.n_components)]
        covariances = [np.identity(self._n, dtype=np.float64) for _ in range(self.n_components)]

        self._weights = np.array(weights)
        self._means = kmeans.cluster_centers_
        self._covariances = np.array(covariances)

        # self._weights_array, _ = np.broadcast_arrays(self._weights, np.zeros(shape=(self._m, self.n_components)))
        return

    @staticmethod
    def gaussian(X, mu, cov):
        """
        多元高斯密度函数
        https://www.ituring.com.cn/book/miniarticle/203200
        :param X: ndarray, 形状为 (m, n)
        :param mu: ndarray, 形状为 (n,)
        :param cov: ndarray, 形状为 (n, n)
        :return: ndarray, 形状为 (m,). 表示 X 中每个样本属于此高斯分布的概率.
        """
        _, n = X.shape
        diff = (X - np.expand_dims(mu, axis=0))
        result = np.diagonal(1 / ((2 * np.pi)**(n/2)*np.linalg.det(cov)**0.5) *
                             np.exp(-0.5*np.dot(np.dot(diff, np.linalg.inv(cov)), diff.T)))
        return result

    def _expectation_step(self):
        """
        计算各高斯分布产生 X 数据的概率, self._expectation_array 形状为 (m, n_components)
        以及数据 X 属于各高斯分布的概率, self._weights_array 形状为 (m, n_components)
        """
        self._expectation_array = np.zeros(shape=(self._m, self.n_components))

        for i in range(self.n_components):
            self._expectation_array[:, i] = self._weights[i] * self.gaussian(self._X, self._means[i], self._covariances[i])
        self._weights_array = self._expectation_array / np.sum(self._expectation_array, axis=1, keepdims=True)
        return

    def _maximization_step(self):
        """
        更新 _weights, _means, _covariances.
        """
        _means = list()
        for i in range(self.n_components):
            # 这里的均值向量不能使用 np.mean 来求, 须注意 np.sum(self._weights_array[:, i]) != self._m
            _mean = np.sum(np.expand_dims(self._weights_array[:, i], axis=1) * self._X, axis=0) / \
                    np.sum(self._weights_array[:, i])
            _means.append(_mean)

        _covariances = list()
        for i in range(self.n_components):
            _X_mean = (self._X - np.expand_dims(self._means[i], axis=0))
            # 求协方差矩阵需注意每个样本的权重
            _covariance = np.dot(_X_mean.T, _X_mean*np.expand_dims(self._weights_array[:, i], axis=1)) / \
                          np.sum(self._weights_array[:, i])
            _covariances.append(_covariance)

        self._means = np.array(_means)
        self._covariances = np.array(_covariances)
        self._weights = np.sum(self._weights_array, axis=0) / np.sum(self._weights_array)
        return

    def fit(self, X):
        self._X = X
        self._m, self._n = self._X.shape
        self._initialize_clusters()

        for i in range(self.max_iter):
            self._expectation_step()
            self._maximization_step()
        return


def demo_sklearn():
    iris = load_iris()
    X = iris.data

    gmm = GaussianMixture(n_components=3, max_iter=50)
    gmm.fit(X)
    print(gmm.weights_)
    print(gmm.means_)
    print(gmm.covariances_)
    return


def demo_gmm():
    iris = load_iris()
    X = iris.data

    gmm = GMM(n_components=3, max_iter=50)
    gmm.fit(X)
    print(gmm.weights_)
    print(gmm.means_)
    print(gmm.covariances_)
    return


if __name__ == '__main__':
    demo_gmm()
