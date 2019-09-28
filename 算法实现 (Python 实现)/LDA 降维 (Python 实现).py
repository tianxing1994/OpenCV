"""
参考链接:
https://blog.csdn.net/z962013489/article/details/79918758

线性判别分析(LDA)
线性判别分析 (Linear Discriminant Analysis, LDA) 是一种经典的线性学习方法, 思路是将两种数据投影到一条直线上,
使这两种数据之间尽可能远离, 且同类数据尽可能聚集在一起.

步骤:
1. 各类别的数据分别减去其均值, 得到 X_mean
2. 将数据集中的各类别样本分开成单独的样本集
3. 求全局的均值向量, 及各类别的均值向量. (即求样本的平均值)
4. 求各类别的协方差矩阵, 再求和得到 Sw. 作为类内离散度衡量指标.
5. 求类间离散度 Sb, 将各类别的均值向量减去数据集的全局均值向量, 该向量与其转置向量相乘得到广播后的方阵.
   作为类间离散度的协方差矩阵. 各类别的类间离散度协方差矩阵乘以各类的样本数作为其权重. 得到 Sb
6. 我们要类间离散度更大, 类内离散度更小, 所以用 Sb/Sw, 因为 Sw 是一个矩阵, 所以需要求其逆矩阵. 得到方阵 w
7. 求方阵 w 的特征向量, 取特征值最大的 n 个特征向量.
8. 对 X_mean 应用 n 个特征向量, 即得到降维后的数据集.
注: 我不知道, 为什么 Sb 要那样求, 也不知道为什么是求方阵 w 的特征向量就可以实现目标.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def lda(x, y, n):
    """
    LDA 降维
    """
    # 获取有哪些类别.
    classify = np.unique(y)

    # 类别的数量
    l = len(classify)

    # 将各类别数据分离
    x_s = [x[y == c] for c in classify]

    # 求各类别的均值向量.
    mju = [np.mean(x, axis=0) for x in x_s]

    # 求全局均值向量. mg.shape=(n,)
    mg = np.mean(x, axis=0)

    # 求类内离散度. 求各类别减去其均值向量后求协方差矩阵, 并求和.
    sw = np.sum([np.dot((x_s[i]-mju[i]).T, (x_s[i]-mju[i])) for i in range(l)], axis=0)

    # 求类间离散度.
    sb = np.sum([len(mju[i]) * np.expand_dims(mju[i]-mg, axis=1) * np.expand_dims(mju[i]-mg, axis=0) for i in range(l)], axis=0)

    # 求损失函数 w, 即 sb/sw.
    # np.mat(Sw).I 求逆矩阵. 为 mat 类型
    w = np.array(np.dot(np.mat(sw).I, sb))
    # w = np.dot(np.linalg.inv(sw), sb)

    # 求特征值, 特征向量. eigVal.shape=(4,), eigVect.shape=(4, 4). eigVect 中每一列是一个特征向量
    eig_val, eig_vect = np.linalg.eig(w)
    # print(eig_val)
    # 对特征值排序, 并取出特征值最大的 n 个特征向量.
    eig_val_index = np.argsort(eig_val)
    eig_val_index = eig_val_index[-1:-(n+1):-1]
    max_eig_vect = eig_vect[:, eig_val_index]

    # 用特征向量对 data_x_mean 进行变换. 得到结果
    result = np.dot(x, max_eig_vect)
    return result


def lda_demo():
    iris = load_iris()
    x = iris.data
    y = iris.target
    x_new = lda(x, y, 2)

    plt.figure(2)
    plt.scatter(x_new[:, 0], x_new[:, 1], marker='o', c=y)
    plt.show()
    return


def sklearn_lda_demo():
    """sklearn LDA 降维
    http://lijiancheng0614.github.io/scikit-learn/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
    """
    iris = load_iris()
    x = iris.data
    y = iris.target

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(x, y)
    x_new = lda.transform(x)

    plt.figure(2)
    plt.scatter(x_new[:, 0], x_new[:, 1], marker='o', c=y)
    plt.show()
    return


if __name__ == '__main__':
    sklearn_lda_demo()
