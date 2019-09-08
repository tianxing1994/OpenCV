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
        _covariances = list()
        for i in range(self.n_components):
            _mean = np.mean(np.expand_dims(self._weights_array[:, i], axis=1) * self._X, axis=0)
            _means.append(_mean)

            _covariance = np.zeros(shape=(self._n, self._n))
            n_i = np.sum(self._weights_array, axis=0)[i]
            # 更新协方差矩阵. (它的协方差矩阵跟我想像的不一样啊).
            #TODO: 有问题啊, 需要改啊.
            for j in range(self._m):
                diff = (self._X[j] - self._means[i]).reshape(-1, 1)
                _covariance += self._weights_array[j, i] * np.dot(diff, diff.T)
            _covariance /= n_i
            _covariances.append(_covariance)

        self._means = np.array(_means)
        self._covariances = np.array(_covariances)
        self._weights = np.sum(self._weights_array, axis=0) / np.sum(self._weights_array)
        return

    def fit(self, X):
        self._X = X
        self._m, self._n = self._X.shape
        self._initialize_clusters2()

        for i in range(self.max_iter):
            self._expectation_step()
            self._maximization_step()
        return


def demo_sklearn():
    iris = load_iris()
    X = iris.data

    n_clusters = 3
    n_epochs = 50

    gmm = GaussianMixture(n_components=n_clusters, max_iter=50)
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
