"""
参考链接:
https://blog.csdn.net/danliwoo/article/details/82731157
"""
import math
import random


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
    print(Z)
    print(X)
    return


if __name__ == '__main__':
    observation_demo()




















