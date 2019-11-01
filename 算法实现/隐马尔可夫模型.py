"""
参考链接:
https://blog.csdn.net/danliwoo/article/details/82731157

备注: 这个东西还没弄懂. 暂时不想看了.
"""
import math
import random

import numpy as np


def generate(rate):
    """
    带权重的随机选择.
    该函数随机返回列表范围内的一个索引, 列表中对应位置的值为该索引的权重.
    :param rate: 列表, 列表中对应位置的值为该索引的权重.
    :return:
    """
    r = random.random()
    sum = 0
    for i in range(len(rate)):
        sum += rate[i]
        if r <= sum:
            return i
    return len(rate) - 1


def generate_demo():
    distribution = [0.4, 0.1, 0.5]
    count = [0] * len(distribution)
    for i in range(100000):
        rd = generate(distribution)
        count[rd] += 1
    print(count)
    return


def observation(T, S, H, A, B, pi):
    """
    根据给定的条件, 产生一组隐状态值和观测值.
    :param T: 观测值个数
    :param S: 观测的状态集
    :param H: 隐变量的状态集
    :param A: 隐状态间的转移矩阵
    :param B: 隐状态到观测的发射矩阵
    :param pi: 初始状态概率向量
    :return: Z, X. Z 是隐状态的转换序列, X 是观测值的转换序列.
    """
    # 随机选择一个盒子的索引
    z = generate(pi)
    # 根据选择的盒子, 取一个球.
    x = S[generate(B[z])]
    # Z. 列表, 存放盒子的名字.
    Z = [H[z]]
    # X. 列表, 存放取出的球的颜色
    X = [x]
    for t in range(1, T):
        z = generate(A[z])
        x = S[generate(B[z])]
        Z.append(H[z])
        X.append(x)
    return Z, X


def observation_demo():
    # T, 取 10 次.
    T = 10
    # S, 盒子中有红球和白球.
    S = ['red', 'white']
    # H, 有 4 个盒子.
    H = ['box1', 'box2', 'box3', 'box4']
    # A, 计算各盒子被选中后, 下一次随机选择盒子时, 四个盒子分别被选中的概率.
    A = [[0, 1, 0, 0],
         [0.3, 0, 0.7, 0],
         [0, 0.4, 0, 0.6],
         [0, 0, 0.6, 0.4]]
    # B, 每个盒子中取出红/白球的概率.
    B = [[0.5, 0.5],
         [0.3, 0.7],
         [0.6, 0.4],
         [0.4, 0.6]]
    # pi, 初始, 四个盒子分别被选中的概率.
    pi = [0.4, 0.1, 0.25, 0.25]
    Z, X = observation(T, S, H, A, B, pi)
    return


def calc_alpha(X, T, S, H, A, B, pi):
    """
    前向概率.
    给定观测序列 X. 推测隐状态的转换关系.
    1. 先求第一个观测状态 X_{1} 由各隐状态转换而来的概率.
    2. 根据前一次各隐状态发生观测状态 X_{i-1} 的概率, 推测这一次各隐状态发生的概率 sum.
    3. 由这一次各隐状态发生的概率 sum 和这一次的观测状态 X_{i}, 求这一次的观测值由各隐状态发生而来的概率.
    4. 迭代上述过程, 求出在本次观测状态由各隐状态发生而来的概率. 存入 alpha 列表.
    :return: alpha. 由前向后推算, 各观测状态由各隐状态发生而来的概率. alpha 列表应长为 T, 其中每一个元素长为 H 的长度.
    """
    N = len(H)

    # 计算第一个观测值来自于各盒子的概率.
    ap = list()
    for i in range(N):
        ap.append(pi[i] * B[i][S.index(X[0])])

    alpha = [ap]    # 如: [[0.2, 0.4, 0.32, 0.15]]
    for t in range(1, T):
        ap = list()
        for i in range(N):
            # sum, 这一次的隐状态是 i, 则当前为止的观测事件发生的可能性.
            sum = 0
            for j in range(N):
                sum += alpha[t-1][j] * A[j][i]
            ap.append(sum * B[i][S.index(X[t])])
        alpha.append(ap)
    return alpha


def calc_alpha_demo():
    """
    :return:
    [[2.00000000e-01 3.00000000e-02 1.50000000e-01 1.00000000e-01]
     [4.50000000e-03 1.82000000e-01 3.24000000e-02 7.80000000e-02]
     [2.73000000e-02 5.23800000e-03 1.04520000e-01 2.02560000e-02]
     [7.85700000e-04 4.83756000e-02 6.32808000e-03 4.24886400e-02]
     [7.25634000e-03 2.32185240e-03 2.37424416e-02 1.24753824e-02]
     [3.48277860e-04 1.17273216e-02 3.64421045e-03 1.15413708e-02]
     [1.75909825e-03 5.41788612e-04 9.08036856e-03 2.72122983e-03]
     [8.12682918e-05 1.61737370e-03 1.20719395e-03 2.61468523e-03]
     [2.42606055e-04 3.94902112e-04 1.08038909e-03 1.06211428e-03]
     [5.92353167e-05 4.72333184e-04 3.65480018e-04 6.43847500e-04]]
    """
    T = 10
    S = ['red', 'white']
    H = ['box1', 'box2', 'box3', 'box4']
    A = [[0, 1, 0, 0],
         [0.3, 0, 0.7, 0],
         [0, 0.4, 0, 0.6],
         [0, 0, 0.6, 0.4]]
    B = [[0.5, 0.5],
         [0.3, 0.7],
         [0.6, 0.4],
         [0.4, 0.6]]
    pi = [0.4, 0.1, 0.25, 0.25]
    _, X = observation(T, S, H, A, B, pi)
    alpha = calc_alpha(X, T, S, H, A, B, pi)
    print(np.array(alpha))
    return


def calc_beta(X, T, S, H, A, B, pi):
    """
    后向概率
    :return: beta. 由后向前推算, 各观测状态由各隐状态发生而来的概率. beta 列表应长为 T, 其中每一个元素长为 H 的长度.
    """
    # 隐状态的个数.
    N = len(H)
    bt = [1] * N

    beta = [bt]
    for t in range(T-2, -1, -1):
        bt = list()

        for i in range(N):
            # sum, 这一次隐状态为 i, 则后续观测事件发生的概率.
            sum = 0
            for j in range(N):
                # A[i][j], 这一次是 i, 则后一次是 j 的概率.
                # B[j][S.index(X[t+1])], 后一次为 j 时, 产生后一次的观测值 X[t+1] 的概率.
                # beta[0][j], 后一次发生的可能性是.
                sum += A[i][j] * B[j][S.index(X[t+1])] * beta[0][j]
            bt.append(sum)
        beta.insert(0, bt)
    return beta


def calc_beta_demo():
    """
    :return:
    [[7.16966158e-04 1.25461036e-03 8.24604673e-04 1.16390970e-03]
     [2.10012509e-03 2.38988719e-03 2.23712285e-03 2.24090921e-03]
     [5.74736869e-03 3.00017871e-03 5.45636389e-03 3.88075782e-03]
     [6.18930828e-03 8.21052670e-03 7.39922309e-03 8.77060115e-03]
     [1.20503160e-02 2.06310276e-02 1.52451888e-02 2.05145824e-02]
     [3.20382000e-02 4.01677200e-02 3.76792800e-02 4.34377600e-02]
     [5.88600000e-02 1.06794000e-01 7.46160000e-02 1.03600000e-01]
     [1.71000000e-01 1.96200000e-01 1.93200000e-01 2.12800000e-01]
     [3.00000000e-01 5.70000000e-01 3.60000000e-01 5.20000000e-01]
     [1.00000000e+00 1.00000000e+00 1.00000000e+00 1.00000000e+00]]
    """
    T = 10
    S = ['red', 'white']
    H = ['box1', 'box2', 'box3', 'box4']
    A = [[0, 1, 0, 0],
         [0.3, 0, 0.7, 0],
         [0, 0.4, 0, 0.6],
         [0, 0, 0.6, 0.4]]
    B = [[0.5, 0.5],
         [0.3, 0.7],
         [0.6, 0.4],
         [0.4, 0.6]]
    pi = [0.4, 0.1, 0.25, 0.25]
    _, X = observation(T, S, H, A, B, pi)
    beta = calc_beta(X, T, S, H, A, B, pi)
    print(np.array(beta))
    return


def forword_backword(alpha, beta, t, T, S, H, A, B, pi):
    """
    :return: 0.0022610125891393863
    """
    if t < 0 or t >= T:
        return 0
    sum = 0
    N = len(H)
    for i in range(N):
        sum += alpha[t][i] * beta[t][i]
    return sum


def forword_backword_demo():
    """
    不论 t 的取值是什么, 最后算出来的观测概率都是一样的.
    为什么要大费周章算第 t 个观测的情况.
    :return:
     0 0.000632723772630
     1 0.000632723772630
     2 0.000632723772630
     3 0.000632723772630
     4 0.000632723772630
     5 0.000632723772630
     6 0.000632723772630
     7 0.000632723772630
     8 0.000632723772630
     9 0.000632723772630
    """
    T = 10
    S = ['red', 'white']
    H = ['box1', 'box2', 'box3', 'box4']
    A = [[0, 1, 0, 0],
         [0.3, 0, 0.7, 0],
         [0, 0.4, 0, 0.6],
         [0, 0, 0.6, 0.4]]
    B = [[0.5, 0.5],
         [0.3, 0.7],
         [0.6, 0.4],
         [0.4, 0.6]]
    pi = [0.4, 0.1, 0.25, 0.25]

    _, X = observation(T, S, H, A, B, pi)
    alpha = calc_alpha(X, T, S, H, A, B, pi)
    beta = calc_beta(X, T, S, H, A, B, pi)

    for t in range(T):
        print("{:2d}".format(t), "{:.15f}".format(forword_backword(alpha, beta, t, T, S, H, A, B, pi)))
    return


def viterbi(A, B, pi, O):
    """
    维特比算法, 根据观测值预测隐状态值.
    :param A: 隐状态间的转移矩阵
    :param B: 隐状态到观测的发射矩阵
    :param pi: 初始状态概率向量
    :param O: 用索引表示的观测值状态序列.
    :return:
    """
    n = len(O)
    sigma_1 = pi * B[:, O[0]]
    sigma_list = list()
    sigma_list.append(sigma_1)

    # 计算 sigma
    for i in range(1, n):
        sigma = np.max(A * np.expand_dims(sigma_list[-1], axis=1) * B[:, O[i]], axis=0)
        sigma_list.append(sigma)

    # 最优路径的终点隐状态 i_T.
    i_T = np.argmax(sigma_list[-1])

    # 由最优路径的终点 i_T, 逆向求 I.
    I = list()
    I.append(i_T)
    for i in range(n-2, -1, -1):
        sigma = sigma_list[i]
        I_t = np.argmax((A * np.expand_dims(sigma, axis=1))[:, I[-1]])
        I.append(I_t)
    result = list(reversed(I))
    return result


def viterbi_demo():
    T = 10
    S = ['red', 'white']
    H = ['box1', 'box2', 'box3', 'box4']
    A = np.array([[0, 1, 0, 0],
         [0.3, 0, 0.7, 0],
         [0, 0.4, 0, 0.6],
         [0, 0, 0.6, 0.4]])
    B = np.array([[0.5, 0.5],
         [0.3, 0.7],
         [0.6, 0.4],
         [0.4, 0.6]])
    pi = np.array([0.4, 0.1, 0.25, 0.25])
    Z, X = observation(T, S, H, A, B, pi)

    # 将文字表示的 X 转换为数字
    X_ = np.array([S.index(x) for x in X])

    result_index = viterbi(A, B, pi, X_)

    # 将数字表示的隐状态转换为文字
    result = [H[index] for index in result_index]

    print(X)
    # 真实的隐状态
    print(Z)
    # 预测的隐状态
    print(result)
    return


def normalization(distribution):
    sum = np.sum(distribution)
    if sum == 0:
        return distribution
    result = np.array([x/sum for x in distribution])
    return result


def predict(T, S, H, A, B, pi, X, t):
    """
    预测观测序列中缺失位置 t 的可能值.
    :param T: 观测值个数
    :param S: 观测的状态集
    :param H: 隐变量的状态集
    :param A: 隐状态间的转移矩阵
    :param B: 隐状态到观测的发射矩阵
    :param pi: 初始状态概率向量
    :param X: 观测序列
    :param t: 需要预测的值在观测序列中的索引
    :return:
    """
    alpha = calc_alpha(X, T, S, H, A, B, pi)
    beta = calc_beta(X, T, S, H, A, B, pi)
    N = len(H)
    pd = []
    for sk in S:
        X[t] = sk
        if t == 0:
            for i in range(N):
                alpha[0][i] = pi[i]*B[i][S.index(X[0])]
        else:
            for i in range(N):
                alpha[t][i] = 0
                for j in range(N):
                    alpha[t][i] += alpha[t-1][j]*A[j][i]
                alpha[t][i] *= B[i][S.index(X[t])]
        pd.append(forword_backword(alpha, beta, t, T, S, H, A, B, pi))
    print(pd)
    print('after normalization: ', normalization(pd))
    theta = pd.index(max(pd))
    return S[theta]


def predict_demo():
    T = 10
    S = ['red', 'white']
    H = ['box1', 'box2', 'box3', 'box4']
    A = np.array([[0, 1, 0, 0],
         [0.3, 0, 0.7, 0],
         [0, 0.4, 0, 0.6],
         [0, 0, 0.6, 0.4]])
    B = np.array([[0.5, 0.5],
         [0.3, 0.7],
         [0.6, 0.4],
         [0.4, 0.6]])
    pi = np.array([0.4, 0.1, 0.25, 0.25])
    Z, X = observation(T, S, H, A, B, pi)

    result = predict(T, S, H, A, B, pi, X, 3)

    print(X)
    print(result)
    return


if __name__ == '__main__':
    predict_demo()




















