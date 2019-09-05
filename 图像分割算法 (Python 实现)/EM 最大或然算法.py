"""
参考链接:
http://www.hankcs.com/ml/em-algorithm-and-its-generalization.html
http://91up.cc/72/120808.html

概念:
1. 假设有关于两个硬币 A/B 的数据如下:
B: [1, 0, 0, 0, 1, 1, 0, 1, 0, 1]
A: [1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
A: [1, 0, 1, 1, 1, 1, 1, 0, 1, 1]
B: [1, 0, 1, 0, 0, 0, 1, 1, 0, 0]
A: [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]

其中分别表示, A 硬币连掷 10 次, 出现正面 1, 反面 0 的记录.
由以上数据, 我们可以计算出:
A 掷出正面的概率, P(1|A) = 0.8 = 24/(24+6)
B 掷出正面的概率, P(1|B) = 0.45 = 9/(9+11)

2. 同样是上面的数据, 但是如果我们不知道哪些组分别是 A/B 硬币掷出的, 则应该如何计算 A/B 硬币掷出正面的概率呢.
(1) [1, 0, 0, 0, 1, 1, 0, 1, 0, 1]
(2) [1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
(3) [1, 0, 1, 1, 1, 1, 1, 0, 1, 1]
(4) [1, 0, 1, 0, 0, 0, 1, 1, 0, 0]
(5) [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]

EM 算法:
1. 首先可假设 A/B 硬币掷出正面的概率为:  P(1|A) = 0.6, P(1|B) = 0.5.
2. 计算在 P(1|A) = 0.6, P(1|B) = 0.5 的情况下, 以上 5 组数据, 分别是 A/B 硬币出的概率.

(1) => 5正5反.
P(1|A)^5 * P(0|A)^5 = 0.6^5 * 0.4^5 = 0.0007962624; P(1|B)^5 * P(0|B)^5 = 0.5^5 * 0.5^5 = 0.0009765625
P(A) = 0.0007962624 / (0.0007962624 + 0.0009765625) = 0.45
P(B) = 1 - P(A) = 0.55

(2) => 9正1反.
P(A) = P(1|A)^9*P(0|A)^1 / (P(1|A)^9*P(0|A)^1 + P(1|B)^9*P(0|B)^1) = 0.80
P(B) = 1 - P(A) = 0.20

(3) => 8正2反. P(A)=0.73, P(B)=0.27
(4) => 4正6反. P(A)=0.35, P(B)=0.65
(5) => 7正3反. P(A)=0.65, P(B)=0.35

3. 已知各组数据分别由 A/B 硬币掷出的概率. 则我们可由此计算出 A/B 硬币分别掷出正面的概率:
P(1|A) = (0.45*5+0.8*9+0.73*8+0.35*4+0.65*7) / ((0.45+0.8+0.73+0.35+0.65)*10) = 0.71
P(1|B) = (0.55*5+0.2*9+0.27*8+0.65*4+0.35*7) / ((0.55+0.2+0.27+0.65+0.35)*10) = 0.58

4. 一开始, 我们假设 P(1|A)=0.6, P(1|B)=0.5, 根据数据推算觉得 P(1|A)=0.71, P(1|B)=0.58 更合适.
则我们将 P(1|A)=0.71, P(1|B)=0.58 代入, 迭代 2, 3 步. 直到一定条件再停下来.
最后我们得到的 P(1|A), P(1|B) 即是结果.
P(1|A)=0.8, P(1|B)=0.52

"""
import math
import numpy as np


def em_single(priors, observations):
    """
    :param priors: [theta_a, theta_b]
    :param observations: m*n matrix.
    :return: [new_theta_a, new_theta_b]
    """
    counts = {"a": {'1': 0, '0': 0}, "b": {'1': 0, '0': 0}}
    theta_a = priors[0]
    theta_b = priors[1]

    for observation in observations:
        len_observation = len(observation)
        num_heads = observation.sum()
        num_tails = len_observation - num_heads
        contribution_a = pow(theta_a, num_heads)*pow(1-theta_a, num_tails)
        contribution_b = pow(theta_b, num_heads)*pow(1-theta_b, num_tails)
        weight_a = contribution_a / (contribution_a + contribution_b)
        weight_b = contribution_b / (contribution_a + contribution_b)
        # 更新在当前参数下 A, B 硬币掷出正反面的次数
        counts['a']['1'] += weight_a * num_heads
        counts['a']['0'] += weight_a * num_tails
        counts['b']['1'] += weight_b * num_heads
        counts['b']['0'] += weight_b * num_tails

    new_theta_a = counts['a']['1'] / (counts['a']['1'] + counts['a']['0'])
    new_theta_b = counts['b']['1'] / (counts['b']['1'] + counts['b']['0'])
    return new_theta_a, new_theta_b


def em(observations, prior, tol=1e-6, iterations=10000):
    """
    EM 算法
    :param observations: 观测数据
    :param prior: 初始值
    :param tol: 迭代结束阈值
    :param iterations: 最大迭代次数
    :return: 局部最优的模型参数
    """
    iteration = 0
    while iteration < iterations:
        new_prior = em_single(prior, observations)
        delta_change = np.abs(prior[0] - new_prior[0])
        if delta_change < tol:
            break
        else:
            prior = new_prior
            iteration += 1
    return new_prior, iteration


if __name__ == '__main__':
    observations = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                             [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                             [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                             [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                             [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])

    pa1 = 0.6
    pb1 = 0.5

    result = em(observations, (0.6, 0.5))
    print(result)
