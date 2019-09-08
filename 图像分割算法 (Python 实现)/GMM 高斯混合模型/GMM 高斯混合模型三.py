"""
参考链接:
https://blog.csdn.net/weixin_42137700/article/details/90762583

这个人的代码真的很棒. 和 sklearn GaussianMixture 的结果一样, 还做了 gif 动图.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Ellipse
from PIL import Image
import imageio


def gaussian(X, mu, cov):
    """
    多元高斯密度函数
    https://www.ituring.com.cn/book/miniarticle/203200
    :param X: ndarray, 形状为 (m, n)
    :param mu: ndarray, 形状为 (1, n)
    :param cov: ndarray, 形状为 (n, n)
    :return: ndarray, 形状为 (m, 1). 表示 X 中每个样本属于此高斯分布的概率.
    """
    _, n = X.shape
    diff = (X - mu).T
    result = np.diagonal(1 / ((2*np.pi)**(n/2)*np.linalg.det(cov)**0.5) *
                         np.exp(-0.5*np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1)
    return result


def initialize_clusters(X, n_clusters):
    """
    初始化均值向量, 协方差矩阵, 权重.
    此处, 各类的权重 pi_k 取平均值; 协方差矩阵 cov_k 取单位矩阵; 均值向量取 KMeans 聚类中心.
    :param X: ndarray, 形状为: (m, n).
    :param n_clusters: int, 指定 KMeans 聚类的中心数量.
    :return: 包含字典的列表, 长度为 n_clusters. [{'pi_k': 1/n_clusters, 'mu_k': mu_i, 'cov_k': cov_i}, ...]
    """
    clusters = list()

    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    mu_k = kmeans.cluster_centers_
    for i in range(n_clusters):
        clusters.append({
            'pi_k': 1.0 / n_clusters,
            'mu_k': mu_k[i],
            'cov_k': np.identity(X.shape[1], dtype=np.float64)
        })
    return clusters


def expectation_step(X, clusters):
    """
    计算各高斯分布产生 X 数据的概率之和 totals, 形状为 (m, 1). 及数据属于各高斯分布的概率 gamma_nk, 形状为 (m, 1).
    :param X: ndarray, 形状为: (m, n).
    :param clusters: [{'pi_k': 1/n_clusters, 'mu_k': mu_i, 'cov_k': cov_i}, ...]
    :return: 修改传入 clusters 的内容.
    """
    totals = np.zeros(shape=(X.shape[0], 1), dtype=np.float64)
    for cluster in clusters:
        pi_k = cluster['pi_k']
        mu_k = cluster['mu_k']
        cov_k = cluster['cov_k']
        gamma_nk = (pi_k * gaussian(X, mu_k, cov_k)).astype(np.float64)

        # for i in range(X.shape[0]):
        #     totals[i] += gamma_nk[i]
        totals += gamma_nk

        cluster['gamma_nk'] = gamma_nk
        cluster['totals'] = totals
    for cluster in clusters:
        cluster['gamma_nk'] /= cluster['totals']
    return


def maximization_step(X, clusters):
    """
    更新 pi_k, mu_k, cov_k.
    :param X:
    :param clusters:
    :return:
    """
    N = float(X.shape[0])

    for cluster in clusters:
        gamma_nk = cluster['gamma_nk']
        cov_k = np.zeros(shape=(X.shape[1], X.shape[1]))

        N_k = np.sum(gamma_nk, axis=0)
        pi_k = N_k / N
        mu_k = np.sum(gamma_nk * X, axis=0) / N_k

        for j in range(X.shape[0]):
            diff = (X[j] - mu_k).reshape(-1, 1)
            cov_k += gamma_nk[j] * np.dot(diff, diff.T)
        cov_k /= N_k

        cluster['pi_k'] = pi_k
        cluster['mu_k'] = mu_k
        cluster['cov_k'] = cov_k
    return


def get_likelihood(X, clusters):
    likelihood = list()
    #todo: 我认为这里不应该使用 cluster['totals']. 应直接对各类别的概率取对数再求和, 而不是先在各类别相加后再取对数求和.
    sample_likelihoods = np.log(np.array([cluster['totals'] for cluster in clusters]))
    return np.sum(sample_likelihoods), sample_likelihoods


def train_gmm(X, n_clusters, n_epochs):
    clusters = initialize_clusters(X, n_clusters)
    likelihoods = np.zeros(shape=(n_epochs,))
    scores = np.zeros(shape=(X.shape[0], n_clusters))
    history = list()
    for i in range(n_epochs):
        clusters_snapshot = list()

        for cluster in clusters:
            clusters_snapshot.append({
                'mu_k': cluster['mu_k'].copy(),
                'cov_k': cluster['cov_k'].copy(),
            })
        history.append(clusters_snapshot)
        expectation_step(X, clusters)
        maximization_step(X, clusters)

        likelihood, sample_likelihoods = get_likelihood(X, clusters)
        likelihoods[i] = likelihood
        print('Epoch: ', i+1, 'Likelihood: ', likelihood)

    for i, cluster in enumerate(clusters):
        scores[:, i] = np.log(cluster['gamma_nk']).reshape(-1)
    return clusters, likelihoods, scores, sample_likelihoods, history


def create_cluster_animation(X, history, scores):
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111)
    color_set = ['blue', 'red', 'black']
    images = list()

    for j, clusters in enumerate(history):
        index = 0
        if j % 3 != 0:
            continue
        plt.cla()
        for cluster in clusters:
            mu = cluster['mu_k']
            cov = cluster['cov_k']
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            vx, vy = eigenvectors[:, 0][0], eigenvectors[:, 0][1]
            theta = np.arctan2(vy, vx)

            color = colors.to_rgba(color_set[index])

            for cov_factor in range(1, 4):
                ell = Ellipse(xy=mu,
                              width=np.sqrt(eigenvalues[0]) * cov_factor*2,
                              height=np.sqrt(eigenvalues[1]) * cov_factor*2,
                              angle=float(np.degrees(theta)),
                              linewidth=2)
                ell.set_facecolor(color=(color[0], color[1], color[2], 1.0/(cov_factor*4.5)))
                ax.add_artist(ell)
            ax.scatter(cluster['mu_k'][0], cluster['mu_k'][1], c=color_set[index], s=1000, marker='+')
            index += 1

        for i in range(X.shape[0]):
            ax.scatter(X[i, 0], X[i, 1], c=color_set[int(np.argmax(scores[i]))], marker='o')

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1]+(3,))
        images.append(image)

    kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
    imageio.mimsave('./gmm.gif', images, fps=1)
    plt.show(Image.open('gmm.gif').convert('RGB'))
    return


def demo_gaussian():
    x0 = np.array([[0.05, 1.413, 0.212],
                   [0.85, -0.3, 1.11],
                   [11.1, 0.4, 1.5],
                   [0.27, 0.12, 1.44],
                   [88, 12.33, 1.44]])
    mu = np.mean(x0, axis=0)
    cov = np.dot((x0 - mu).T, x0-mu) / x0.shape[0]

    y = gaussian(x0, mu=mu, cov=cov)
    print(y)
    return


def demo_train_gmm():
    iris = load_iris()
    X = iris.data

    n_clusters = 3
    n_epochs = 50
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(X, n_clusters, n_epochs)

    plt.figure(figsize=(10, 10))
    plt.title('Log-Likelihood')
    plt.plot(np.arange(1, n_epochs + 1), likelihoods)
    plt.show()
    return


def demo_compare_with_sklearn():
    iris = load_iris()
    X = iris.data

    n_clusters = 3
    n_epochs = 50
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(X, n_clusters, n_epochs)

    gmm = GaussianMixture(n_components=n_clusters, max_iter=50).fit(X)
    gmm_scores = gmm.score_samples(X)
    print('Means by sklearn:', gmm.means_)
    print('Means by our implementation:', np.array([cluster['mu_k'].tolist() for cluster in clusters]))
    print('Scores by sklearn:', gmm_scores[0:20])
    print('Scores by our implementation:', sample_likelihoods.reshape(-1)[0:20])
    return


def demo_create_cluster_animation():
    iris = load_iris()
    X = iris.data
    n_clusters = 3
    n_epochs = 50
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(X, n_clusters, n_epochs)

    create_cluster_animation(X, history, scores)
    return


if __name__ == '__main__':
    demo_compare_with_sklearn()
