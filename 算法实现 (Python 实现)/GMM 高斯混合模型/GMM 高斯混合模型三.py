"""
参考链接:
https://blog.csdn.net/qq_30091945/article/details/81134598

现在的代码只能表示原理, 和 sklearn 的 GaussianMixture 结果相差很大呀.
"""
import numpy as np
from sklearn.datasets import load_iris
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn import mixture


def gaussian(x, mean, cov):
    """
    多元高斯概率密度函数. 还不知道是怎么推出来的.
    https://www.ituring.com.cn/book/miniarticle/203200
    :param x: 输入向量
    :param mean: 均值向量
    :param cov: 协方差矩阵
    :return: x 的概率
    """
    _, n = cov.shape
    cov_det = np.linalg.det(cov + np.eye(n) * 0.001)
    cov_inv = np.linalg.inv(cov + np.eye(n) * 0.001)
    x_diff = np.expand_dims(x - mean, axis=0)

    # 概率密度函数
    prob = 1.0 / np.power(np.power(2*np.pi, n)*np.abs(cov_det), 0.5) * \
           np.exp(-0.5 * x_diff.dot(cov_inv).dot(x_diff.T))[0][0]
    return prob


def gmm_em(data, k, weights, means, covars):
    """
    用 EM 算法优化 GMM 参数.
    :param data: 形状为 (m, n) 的 ndarray. m 表示样本数量, n 表示样本维度.
    :param k: 高斯分布的个数.
    :param weights: 各类别的权重, k 个高斯分布, 则 weights 的形状应为 (k,)
    :param means: 均值向量. 如 n 维数据, k 个高斯分布, 则 means 的形状应为 (k, n)
    :param covars: 协方差矩阵. 如 n 维数据, k 个高斯分布, 则 covars 的形状应为 (k, n, n).
    :return: 返回各向量属于各分类的概率.
    """
    loglikelyhood = 0
    oldloglikelyhood = 1
    m, n = data.shape
    # gammas 记录每个样本分别属于各类别的概率.
    gammas = np.zeros(shape=(m, k))
    while np.abs(loglikelyhood - oldloglikelyhood) > 1e-2:
        oldloglikelyhood = loglikelyhood
        for i in range(m):
            # probs, 样本在每个高斯分布中的发生概率, 形状为 (1, k).
            probs = np.array([weights[t] * gaussian(data[i], means[t], covars[t]) for t in range(k)])
            gammas[i] = probs / np.sum(probs)

        # 更新每个高斯分布的权重
        # nk = np.sum(gammas, axis=0)
        # weights = nk / m

        # 更新每个高斯分布的均值
        means = np.array([np.mean(np.array(data * np.expand_dims(gammas[:, t], axis=1)), axis=0) for t in range(k)])

        # 更新每个高斯分布的协方差矩阵
        # x_diffs, 形状为 (k, m, n)
        x_diffs = np.array([data - means[t] for t in range(k)])
        covars = np.array([np.dot((x_diffs[t]*np.expand_dims(gammas[:, t], axis=1)).T,
                                  x_diffs[t]*np.expand_dims(gammas[:, t], axis=1)) for t in range(k)])

        loglikelyhood = list()
        for i in range(m):
            # probs2, 样本在每个高斯分布中的发生概率 (更新后的 weights, means, covars), 形状为 (1, k).
            probs2 = np.array([weights[t] * gaussian(data[i], means[t], covars[t]) for t in range(k)])
            log_probs2 = np.log(probs2)
            loglikelyhood.append(log_probs2)
        # loglikelyhood 形状为 (m, k). 表示的是当前 k 个高斯分布的情况下, 当前数据发生的概率. 此值越大越好.
        loglikelyhood = np.sum(np.array(loglikelyhood))
    for i in range(m):
        gammas[i] /= np.sum(gammas[i])
    return np.argmax(gammas, axis=1)


def demo1():
    iris = load_iris()
    label = iris.target
    data = iris.data
    _, n = data.shape

    # 数据预处理
    data = Normalizer().fit_transform(data)

    # 解决画图是中文乱码问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 数据可视化
    plt.scatter(data[:,0],data[:,1],c = label)
    plt.title("Iris数据集显示")
    plt.show()

    # GMM 模型
    k = 3

    weights = np.array([0.3, 0.3, 0.4])
    means = np.array([np.random.rand(n) for _ in range(n)])
    covars = np.array([np.random.rand(n, n) for _ in range(n)])

    y_pre = gmm_em(data, k, weights, means, covars)
    print(y_pre.shape)
    print(np.unique(y_pre))

    print("GMM预测结果：\n", y_pre)
    print("GMM正确率为：\n", accuracy_score(label, y_pre))
    plt.scatter(data[:, 0], data[:, 1], c=y_pre)
    plt.title("GMM结果显示")
    plt.show()
    return


def demo2():
    """sklearn GMM 模型"""
    iris = load_iris()
    label = iris.target
    data = iris.data
    _, n = data.shape

    g = mixture.GaussianMixture(n_components=3)
    g.fit(data)
    print(g.weights_)
    print(g.means_)
    print(g.covariances_)
    return


if __name__ == '__main__':
    demo1()
