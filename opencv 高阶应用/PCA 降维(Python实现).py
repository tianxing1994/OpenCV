"""
参考链接:
https://www.cnblogs.com/mikewolf2002/p/3429711.html
https://github.com/zhaoxingfeng/PCA/blob/master/PCA.py
https://github.com/jimenbian/PCA/blob/master/src/PCA.py


以下给出几个基础命题.
1. 向量的数量积
a·b = |a| |b| cosθ
其中 |a| 代表 a 向量的长度. 当 |b| 为 1 时, a·b 可以理解为 a 向量在 b 向量方向上的投影.

2. 向量的内积.
由于在高维空间中没有直观的 θ 角度. 应此引入向量的内积作为向量的数量积在高维空间中的推广.
a, b 两向量的内积 [a, b] = [a1b1 + a2b2 + a3b3 + ... + anbn] = ∑aibi
‖a‖ = √[a, a] = √∑ai^2 (向量 a 的长度或范数, 欧氏距离).

3. 基变换与坐标变换
设 (α1, α2, ..., αn) 及 (β1, β2, ..., βn) 是线性空间 Vn 中的两个基. 有:
β1 = p11*α1 + p21*α1 + ... + pn1*αn
β2 = p12*α1 + p22*α1 + ... + pn2*αn
...
βn = p1n*α1 + p2n*α1 + ... + pnn*αn

令 P = [[p11, p21, ..., pn1], [p12, p22, ..., pn2], ..., [p1n, p2n, ..., pnn]]
则有: (β1, β2, ..., βn) = (α1, α2, ..., αn) · P
上式称为"基变换公式".

4. 线性变换
设 Vn, Um 分别是 n 维和 m 维线性空间, T 是一个从 Vn 到 Um 的映射, 如果映射 T 满足:
(i) 任给 α1,α2∈Vn (从而 α1+α2∈Vn), 有 T(α1+α2) = T(α1) + T(α2);
(ii) 任给 α∈Vn, λ∈R(从而 λα∈Vn), 有: T(λα) = λT(α)
那么, T 就称为从 Vn 到 Um 的线性映射, 或称为线性变换.

例如:
(y1, y2, ..., ym) = (x1, x2, ..., xn) · [[a11, a21, ..., am1], [a12, a22, ..., am2], ..., [a1n, a2n, ..., amn]]
就确定了一个从 R^n 到 R^m 的映射, 并且是一个线性映射.

5. 期望(均值).
期望是试验中每次可能结果的概率乘以其结果的总和.
E(X) = ∑xipi
对于一组样本, 其期望为其均值(因为 pi=1/n).
E(X) = 1/n∑xi

6. 方差.
方差用于表示一组样本的离散程度. 其计算公式如下:
Var(a) = 1/n∑(ai-μ)^2
其中 μ 代表样本的期望.

7. 协方差.
方差其实是协方差的一种特殊情况. 协方差用于表示一组样本中, 两个特征之间的相关性. 当两个特征完全独立, 不相关时, 其协方差为 0.
如一组数据有 a, b 两个特征值, n 个样本. 则 a,b 特征的协方差计算如下:
Cov(a, b) = 1/n∑(ai-μa)(bi-μb)
其中 μa 代表 a 特征的期望, μb 代表 b 特征的期望.
如果, 我们事先使样本的减去其均值, 并除以 n. 得到 a_, b_ 则以式可以写为:
Cov(a, b) = ∑a_i*b_i

8. 协方差矩阵.
假设有一个矩阵 X:
X = \
[[a11, a12, a13, ..., a1n],
[a21, a22, a23, ..., a2n],
[a31, a32, a33, ..., a3n],
...,
[am1, am2, m33, ..., amn]]

D = np.dot(X, np.transpose(X)) = \
[[∑a1i*a1i, ∑a1i*a2i, ∑a1i*a3i, ..., ∑a1i*ami],
[∑a2i*a1i, ∑a2i*a2i, ∑a2i*a3i, ..., ∑a2i*ami],
[∑a3i*a1i, ∑a3i*a2i, ∑a3i*a3i, ..., ∑a3i*ami],
...,
[∑ani*a1i, ∑ani*a2i, ∑ani*a3i, ..., ∑ani*ami]]

根据 Cov(a, b) = ∑a_i*b_i. 可以看出, 以上等式可以看作是 X 矩阵中每一行(特征)之间的协方差矩阵.
其中对角线上的值是 X 矩阵每一行(特征)的方差, 其它的值是特征之间的协方差.
如果以上等式中除了对角线上的值不为 0, 其它值都为 0. 则说明矩阵 X 中的每一行(特征)相互独立.


PCA 降维的思想:
当我们想将一组样本从 m 个特征降到 n 个特征时,
我们将原来的 m 个特征视为一个 m 维的空间(记作 M).
此时从 M 维空间中选出 n 个完全正交的向量, 做为新的 n 维空间(记作 N),
将原来 M 维空间中的向量投射到 N 维空间中, 用 n 个度表示原来的样本. 则实现了降维.
我们对 N 维空间的 n 个向量选择的要求是:
样本在每一个维度中的投影值的方差都应尽可能地大, 且其 n 个维度之间应是相互独立的(即: 协方差为 0).

PCA 算法步骤:
设有m条n维数据.
1）将原始数据按列组成 n 行 m 列矩阵 X
2）将 X 的每一行（代表一个属性字段）进行零均值化，即减去这一行的均值
3）求出协方差矩阵 1/mXX^T
4）求出协方差矩阵的特征值及对应的特征向量
5）将特征向量按对应特征值大小从上到下按行排列成矩阵，取前 k 行组成矩阵 P
6）Y=PX 即为降维到 k 维后的数据
"""

import numpy as np
from sklearn.decomposition import PCA


# 根据保留多少维特征进行降维
class PCAcomponent(object):
    """
    调用方法.
    pca = PCAcomponent(X, 2)
    # pca = PCApercent(data, 0.98)
    pca.fit()
    print(pca.low_dataMat)
    print(pca.variance_ratio)
    """
    def __init__(self, X, N=3):
        self.X = X
        self.N = N
        self.variance_ratio = []
        self.low_dataMat = []

    def _fit(self):
        X_mean = np.mean(self.X, axis=0)
        dataMat = self.X - X_mean
        # 另一种计算协方差矩阵的方法：dataMat.T * dataMat / dataMat.shape[0]
        # 若rowvar非0，一列代表一个样本；为0，一行代表一个样本
        covMat = np.cov(dataMat, rowvar=False)
        # 求特征值和特征向量，特征向量是按列放的，即一列代表一个特征向量
        eigVal, eigVect = np.linalg.eig(np.mat(covMat))
        eigValInd = np.argsort(eigVal)
        eigValInd = eigValInd[-1:-(self.N + 1):-1]  # 取前N个较大的特征值
        small_eigVect = eigVect[:, eigValInd]  # *N维投影矩阵

        print(dataMat.shape)
        print(small_eigVect.shape)
        print(dataMat)
        print(small_eigVect)

        self.low_dataMat = dataMat * small_eigVect  # 投影变换后的新矩阵
        # reconMat = (low_dataMat * small_eigVect.I) + X_mean  # 重构数据
        # 输出每个维度所占的方差百分比
        [self.variance_ratio.append(eigVal[i] / sum(eigVal)) for i in eigValInd]
        return self.low_dataMat

    def fit(self):
        self._fit()
        return self


# 根据保留多大方差百分比进行降维
class PCApercent(object):
    def __init__(self, X, percentage=0.95):
        self.X = X
        self.percentage = percentage
        self.variance_ratio = []
        self.low_dataMat = []

    # 通过方差百分比选取前n个主成份
    def percent2n(self, eigVal):
        sortVal = np.sort(eigVal)[-1::-1]
        percentSum, componentNum = 0, 0
        for i in sortVal:
            percentSum += i
            componentNum += 1
            if percentSum >= sum(sortVal) * self.percentage:
                break
        return componentNum

    def _fit(self):
        X_mean = np.mean(self.X, axis=0)
        dataMat = self.X - X_mean
        covMat = np.cov(dataMat, rowvar=False)
        eigVal, eigVect = np.linalg.eig(np.mat(covMat))
        n = self.percent2n(eigVal)
        eigValInd = np.argsort(eigVal)
        eigValInd = eigValInd[-1:-(n + 1):-1]
        n_eigVect = eigVect[:, eigValInd]
        self.low_dataMat = dataMat * n_eigVect
        [self.variance_ratio.append(eigVal[i] / sum(eigVal)) for i in eigValInd]
        return self.low_dataMat

    def fit(self):
        self._fit()
        return self

# df = pd.read_csv(r'iris.txt', header=None)
# data, label = df[range(len(df.columns) - 1)], df[[len(df.columns) - 1]]
# data = np.mat(data)
# print("Original dataset = {}*{}".format(data.shape[0], data.shape[1]))
# pca = PCAcomponent(data, 3)
# # pca = PCApercent(data, 0.98)
# pca.fit()
# print(pca.low_dataMat)
# print(pca.variance_ratio)


def PCAcomponent_mine(X):
    """
    我实现的 PCAcomponent.
    :param X:
    :return:
    """
    X_mean = X - np.mean(X, axis=0)
    X_mean_m = X_mean / X.shape[0]

    # 求 X 的协方差矩阵
    X_cov = np.dot(X_mean_m.T, X_mean_m)
    # X_cov = np.cov(X, rowvar=False) # rowvar=True, 表示每行是一个特征, 为 False, 则每列是一个特征.

    # 求特征值和特征向量. eigVal.shape=(5, 2). eigVect.shape=(4, 4). 其中每一列代表一个特征向量.
    eigVal, eigVect = np.linalg.eig(X_cov)

    eigValInd = np.argsort(eigVal)

    # 取特征值最大的 2 个特征向量作为新的向量基.
    eigValInd = eigValInd[-1:-(2 + 1):-1]
    small_eigVect = eigVect[:, eigValInd]
    result = np.dot(X_mean, small_eigVect)
    return result


def demo(X):
    """sklearn PCA 降维"""
    pca = PCA(n_components=2)
    result_pca = pca.fit_transform(X)
    print(result_pca)
    return



if __name__ == '__main__':
    X = np.array([[2, 4, 5, 1],
                  [7, 5, 2, 4],
                  [8, 5, 4, 2],
                  [4, 3, 7, 9],
                  [1, 2, 3, 1]])

    result = PCAcomponent_mine(X)
    print(result)