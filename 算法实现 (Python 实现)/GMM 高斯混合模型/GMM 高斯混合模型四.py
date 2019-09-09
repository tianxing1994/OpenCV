"""
参考链接:
https://www.cnblogs.com/lyrichu/p/7814651.html
https://blog.csdn.net/qq_30091945/article/details/81134598

此处的示例针对的是 1 维数据.
"""
import numpy as np


def generate_data(loc_l, scale_l, size_l):
    seq = list()
    for loc, scale, size in zip(loc_l, scale_l, size_l):
        seq.append(np.random.normal(loc=loc, scale=scale, size=size))
    result = np.concatenate(seq=seq, axis=0)
    return result


def gaussian(x, loc, scale):
    """计算均值为 mu, 标准差为 sigma 的正态分布中, x 值所对应的概率."""
    return (1 / np.sqrt(2*np.pi)) * (np.exp(-(x-loc)**2 / (2*scale**2)))


def em1(data_array, k, loc, scale, iterations=10):
    """
    此方法让样本数量的比例也可以训练, 其结果是, 其中一个样本的数量占比特别大, 其它样本数量占比小.
    :param data_array:
    :param k: 如 [0.2, 0.3, 0.5] 的列表, 其值不一定要和为 1. 用于表示各样本的数量比值关系. 也表示目标中心的数量
    :param loc: 如 [1, 5, 10] 的列表, 用于表示各聚类样本的中心.
    :param scale: 如 [2, 3, 2] 的列表, 用于表示各类样本的标准差.
    :param iterations: 指定 EM 迭代的次数.
    :return:
    """
    n = len(k)
    data_num = data_array.size
    gama_array = np.zeros(shape=(n, data_num))
    for iteration in range(iterations):
        for i in range(n):
            for j in range(data_num):
                total = sum([k[t]*gaussian(data_array[j], loc[t], scale[t]) for t in range(n)])
                gama_array[i, j] = k[i]*gaussian(data_array[j], loc[i], scale[i]) / total
        for i in range(n):
            loc[i] = np.sum(gama_array[i] * data_array) / np.sum(gama_array[i])
        for i in range(n):
            scale[i] = np.sqrt(np.sum(gama_array[i]*(data_array - loc[i])**2) / np.sum(gama_array[i]))
        for i in range(n):
            k[i] = np.sum(gama_array[i]) / data_num
    return k, loc, scale


def em2(data_array, k, loc, scale, iterations=10):
    """
    此方法用 k 预先指定各类样本的数量占比, 但 k 不可训练. 只训练 loc, scale 均值和标准差.
    :param data_array:
    :param k: 如 [0.2, 0.3, 0.5] 的列表, 其值不一定要和为 1. 用于表示各样本的数量比值关系. 也表示目标中心的数量
    :param loc: 如 [1, 5, 10] 的列表, 用于表示各聚类样本的中心.
    :param scale: 如 [2, 3, 2] 的列表, 用于表示各类样本的标准差.
    :param iterations: 指定 EM 迭代的次数.
    :return:
    """
    n = len(k)
    data_num = data_array.size
    gama_array = np.zeros(shape=(n, data_num))
    for iteration in range(iterations):
        for i in range(n):
            for j in range(data_num):
                total = sum([k[t]*gaussian(data_array[j], loc[t], scale[t]) for t in range(n)])
                gama_array[i, j] = k[i]*gaussian(data_array[j], loc[i], scale[i]) / total
        for i in range(n):
            loc[i] = np.sum(gama_array[i] * data_array) / np.sum(gama_array[i])
        for i in range(n):
            scale[i] = np.sqrt(np.sum(gama_array[i]*(data_array - loc[i])**2) / np.sum(gama_array[i]))
    return loc, scale


def em3(data_array, n, loc, scale, iterations=10):
    """
    此方法假设各样本的数量是均匀的, 且不可训练. 只训练 loc, scale 均值和标准差.
    :param data_array:
    :param n: 表示目标中心的数量
    :param loc: 如 [1, 5, 10] 的列表, 用于表示各聚类样本的中心.
    :param scale: 如 [2, 3, 2] 的列表, 用于表示各类样本的标准差.
    :param iterations: 指定 EM 迭代的次数.
    :return:
    """
    data_num = data_array.size
    gama_array = np.zeros(shape=(n, data_num))
    for iteration in range(iterations):
        for i in range(n):
            for j in range(data_num):
                total = sum([gaussian(data_array[j], loc[t], scale[t]) for t in range(n)])
                gama_array[i, j] = gaussian(data_array[j], loc[i], scale[i]) / total
        for i in range(n):
            loc[i] = np.sum(gama_array[i] * data_array) / np.sum(gama_array[i])
        for i in range(n):
            scale[i] = np.sqrt(np.sum(gama_array[i]*(data_array - loc[i])**2) / np.sum(gama_array[i]))
    return loc, scale


def demo1():
    loc_l = [1, 5, 10]
    scale_l = [2, 2, 2]
    size_l = [2000, 2000, 2000]
    data_array = generate_data(loc_l, scale_l, size_l)

    k = [1, 1, 1]
    loc = [1, 2, 3]
    sigma = [2, 2, 2]
    iterations = 20

    k1, mu1, sigma1 = em1(data_array, k, loc, sigma, iterations)
    print(k1, mu1, sigma1)
    return


def demo2():
    # loc_l = [1, 5, 10]
    # scale_l = [2, 2, 2]
    loc_l = [1, 3, 5]
    scale_l = [2, 2, 2]
    size_l = [2000, 5000, 2000]
    data_array = generate_data(loc_l, scale_l, size_l)

    k = [2, 5, 2]
    loc = [1, 2, 3]
    sigma = [2, 2, 2]
    iterations = 20

    loc_ret, sigma_ret = em2(data_array, k, loc, sigma, iterations)
    print(loc_ret, sigma_ret)
    return


def demo3():
    loc_l = [1, 5, 10]
    scale_l = [2, 2, 2]
    # size_l = [2000, 2000, 2000]
    size_l = [1000, 1000, 2000]
    data_array = generate_data(loc_l, scale_l, size_l)

    n = 3
    loc = [1, 2, 3]
    sigma = [2, 2, 2]
    iterations = 20

    loc_ret, sigma_ret = em3(data_array, n, loc, sigma, iterations)
    print(loc_ret, sigma_ret)
    return


if __name__ == '__main__':
    demo2()







































