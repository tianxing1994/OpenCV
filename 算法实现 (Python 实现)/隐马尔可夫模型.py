"""
参考链接:
https://blog.csdn.net/danliwoo/article/details/82731157
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


def viterbi(T, S, H, A, B, pi, X):
    N = len(H)
    sg = list()
    parent = [0]
    # 计算第一个观测值来自于各盒子的概率.
    for i in range(N):
        sg.append(pi[i] * B[i][S.index(X[0])])

    for t in range(1, T):
        sigma = sg
        sg = list()
        pt = list()

        # 前一次为哪个隐状态时, 这一次为 i 隐状态的概率最大.
        for i in range(N):
            maxindex, maxvalue = [-1, 0]
            for j in range(N):
                if sigma[j] * A[j][i] > maxvalue:
                    maxvalue = sigma[j] * A[j][i]
                    maxindex = j

            sg.append(maxvalue * B[i][S.index(X[t])])
            pt.append(maxindex)
        parent.append(pt)

    for i in range(N):
        maxindex, maxvalue = [-1, 0]
        if sigma[i] > maxvalue:
            maxvalue = sigma[i]
            maxindex = i
    parent.append(maxindex)
    return parent


def get_solution(parent, T):
    ind = [parent[T]]
    ret = [H[ind[0]]]
    for t in range(T-1, 0, -1):
        p = parent[t][ind[0]]
        ind.insert(0, p)
        ret.insert(0, H[p])
    return ret


if __name__ == '__main__':
    forword_backword_demo()




















