"""
K-Means 算法最终的聚类效果受初始的聚类中心的影响,
K-Means++ 算法未选择较好的初始聚类中心提供了依据, 但在 K-Means 算法中,
聚类的类别个数 k 仍需要事先指定. 对于类别个数未知的, K-Means 算法和 K-Means++ 算法很难将其进行精确求解. 
MeanShift 算法被提出用于解决聚类个数未知的情况.

参考链接:
介绍的参考链接:
https://blog.csdn.net/IMWTJ123/article/details/88972144

代码的参考链接: (它的实现原理是全局样本根据当前的中心得到的核密度函数的权值计算质心, 作为新的中心, 这好像是不正常的).
https://blog.csdn.net/kwame211/article/details/80226696

MeanShift 聚类算法
一个中心点, 根据其一定范围内的临近样本点计算均值向量, 作为新的中心.
如此迭代, 直到中心不再改变. 此方法可以驱使中心向样本更密集的区域移动.

改进的 Mean Shift:
为了使其临近样本点中距离中心更近的样本对中心的移动具有更大的权重, 提出了核函数, 来根据感兴趣样本与中心的距离为其赋予权重.

备注:
推荐先使用 KMeans 聚类将样本聚类成几个种心点 (n_clusters 值设置得大一点, 多得出几个中心),
将这些中心点作为 Mean Shift 聚类的种子点.

"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


class MeanShift(object):
    def __init__(self, epsilon=1e-5, band_width=0.5, min_quantity=3, bin_seeding=False):
        """
        :param epsilon:
        :param band_width: 指定高维球体的半径.
        :param min_fre: 作为起始质心的球体内最少的样本数目.
        :param bin_seeding: [(ndarray, 0), ..., (ndarray, 0)] 列表.
        """
        self._epsilon = epsilon
        self._band_width = band_width
        self._min_quantity = min_quantity
        self._bin_seeding = bin_seeding
        self._radius2 = self._band_width ** 2

        self._m = None
        self._labels = None
        self._centers = list()
        self._seeds = None

    def _get_seeds(self, data):
        """
        种子点初始化比较麻烦, 需要较少, 但又均匀地分布在样本点上. _get_seeds2 是参考链接中的初始化方法.
        我写了这个, 直接将种子布满.
        不过还是推荐用 Kmeans 多聚类出几个中心点. 然后再从这些点进行均值漂移算法.
        :param data:
        :return:
        """
        m = data.shape[0]
        x = np.linspace(data[:, 0].min(), data[:, 0].max(), np.sqrt(m))
        y = np.linspace(data[:, 1].min(), data[:, 1].max(), np.sqrt(m))
        X, Y = np.meshgrid(x, y)
        seeds_array = np.c_[X.ravel(), Y.ravel()]

        seed_list = list()
        for seed in seeds_array:
            seed_list.append((seed, 0))
        self._seeds = np.array(seed_list)
        return self._seeds

    def _get_seeds2(self, data):
        # 获取可以作为起始质心的点 (seed).
        if self._bin_seeding:
            bin_size = self._band_width
        else:
            bin_size = 1
        seed_list = list()
        seeds_quantity = defaultdict(int)
        for sample in data:
            # 将数据粗粒化, 以防止非常近的样本点都作为起始质心.
            # (对样本进行四舍五入, 则相近的样本会计算出相同的值, 先除以 bin_size, 则相当于将样本空间缩小, ).
            seed = tuple(np.round(sample / bin_size))
            seeds_quantity[seed] += 1
        for seed, quantity in seeds_quantity.items():
            if quantity >= self._min_quantity:
                seed_list.append((np.array(seed), 0))
        if not seed_list:
            raise ValueError('the bin size and min_quantity are not proper.')

        self._seeds = np.array(seed_list)
        return self._seeds

    @property
    def seeds(self):
        if len(self._seeds) == 0:
            raise AttributeError("attribute seeds not exist.")
        result = list()
        for seed in self._seeds:
            result.append(seed[0])
        result = np.array(result)
        return result

    @staticmethod
    def _euclidean_distance(center, sample):
        """
        计算均值点到每个样本点的欧氏距离(平方).
        :param center: ndarray, 形状为 (n,) 中心点.
        :param sample: ndarray, 形状为 (n,) 样本点.
        :return:
        """
        delta = center - sample
        return np.sqrt(np.dot(delta, delta))

    def _gaussian_kernel(self, distance2):
        """
        获取高斯核.
        self._band_width 相当于高斯核函数中的 σ. 即方差. (即, 这里取高斯分布的一个方差内的部分).
        高斯核密度函数: 1 / (delta * np.sqrt(2*np.pi)) * np.exp(-1 * (x - u)**2 / (2*delta**2))
        :param distance2: 标量, 为样本点与中心点的空间距离的平方. 相当于高斯核函数中的 (x-μ)^2
        :return:
        """
        result = 1.0 / self._band_width * (2 * np.pi) ** (-1.0 / 2) * np.exp(- distance2 / (2 * self._band_width ** 2))
        return result

    def _shift_center(self, current_center, data):
        """
        计算下一个均值漂移的中心.
        使用带有权重的样本项献, 先计算每个感兴趣样本与中心的差值作为牵引向量, 为每个牵引向量设置权重. 求和. 得到中心迁移向量 mean_shift
        next_center = center + mean_shift
        得到新的中心.
        因为原本不代有权重计算迁移向量 mean shift 时, 是感兴趣样本的向量差之和, 所以:
        为使感兴趣区域样本的权重之和为感兴趣区域样本的数量 n, 则应取 weight = n * kernel_weight / ∑kernel_weight.
        上式改为:
        mean_shift = ∑(weight * vector) = n * ∑(kernel_weight * vector) / ∑kernel_weight
        :param current_center: (ndarray, int), ndarray 形状为 (n,), int 代表该中心内大概有多少个感兴趣样本.
        :param data: ndarray, 形状为 (m, n)
        :return: next_center, count. 返回新的中心, 和旧的中心里有多少个感兴趣样本.
        """
        count = 0
        kernel_weight_sum = 1e-5
        interested_sample_vector_sum = np.zeros_like(current_center[0])
        for index, sample in enumerate(data):
            distance = self._euclidean_distance(current_center[0], sample)
            if distance <= self._band_width:
                kernel_weight = self._gaussian_kernel(distance**2)
                count += 1
                interested_sample_vector_sum += kernel_weight * (sample - current_center[0])
                kernel_weight_sum += kernel_weight
        # print(count / kernel_weight_sum)
        # mean_shift_vector = count * interested_sample_vector_sum / kernel_weight_sum
        mean_shift_vector = interested_sample_vector_sum / kernel_weight_sum
        # print(mean_shift_vector)
        next_center = current_center[0] + mean_shift_vector
        result = (next_center, count)
        return result

    def _mean_shift_the_final_center(self, current_center):
        while True:
            next_center = self._shift_center(current_center, data)
            delta_distance = np.linalg.norm(next_center[0] - current_center[0], 2)
            if delta_distance < self._epsilon:
                break
            current_center = next_center
        return current_center

    def fit(self, data):
        """
        :param data: ndarray, 形状为 (m, n)
        :return:
        """
        self._m = data.shape[0]
        seeds_list = self._get_seeds(data)
        for seed in seeds_list:
            current_center = seed
            # 进行均值漂移
            final_center = self._mean_shift_the_final_center(current_center)
            # 若该次漂移结束后, 最终的质心与已存在的质心距离小于带宽, 则合并.
            for i in range(len(self._centers)):
                if np.linalg.norm(final_center[0] - self._centers[i][0], 2) < self._band_width:
                    if final_center[1] > self._centers[i][1]:
                        # 两个距离很近的质心, 保留感兴趣样本多的一个.
                        self._centers[i] = final_center
                    break
            else:
                if final_center[1] > self._min_quantity:
                    self._centers.append(final_center)

        result = self._classify(data)
        return result

    def _classify(self, data):
        distance_list = list()
        for i, center in enumerate(self._centers):
            center_distances2 = np.sum(np.power(center[0] - data, 2), axis=1)
            distance_list.append(center_distances2)
        distance_array = np.array(distance_list)

        self._labels = np.argmin(distance_array, axis=0)
        return self._labels

    @property
    def n_clusters(self):
        if len(self._centers) == 0:
            raise AttributeError("attribute n_clusters not exist.")
        result = list()
        for center in self._centers:
            result.append(center[0])
        result = np.array(result)
        return result


if __name__ == '__main__':
    data, target = make_blobs(n_samples=500, centers=5, cluster_std=1.2, random_state=None)

    mean_shift = MeanShift(epsilon=1e-5, band_width=2, min_quantity=10, bin_seeding=True)
    result = mean_shift.fit(data)
    # print(result)
    # print(mean_shift.n_clusters)

    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(mean_shift.seeds[:, 0], mean_shift.seeds[:, 1])
    plt.scatter(mean_shift.n_clusters[:, 0], mean_shift.n_clusters[:, 1])
    plt.show()
