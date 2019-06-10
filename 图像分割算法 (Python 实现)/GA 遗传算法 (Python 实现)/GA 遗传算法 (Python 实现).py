"""
参考链接:
https://blog.csdn.net/czrzchao/article/details/52314455

示例:
求 y = 10 * sin(5x) + 7 * cos(4x) 的最大值.

我的评价:
他这个计算步骤体现了遗传算法的思想和计算步骤.
但是对于 "求 y = 10 * sin(5x) + 7 * cos(4x) 的最大值" 这个问题本身,
其 crossover 和 mutation 函数中的交配和基因突变存在问题.
其交配, 只是将二进制序列随机交换一段, 其结果与直接取一个随机数没有区别.
基因突变, 也是同样.

这里, 我的思路是,
1. 0-1 之间生成 400 个随机数 x, 计算其对应值 y.
2. 求平均值 y_mean, 小于平均值的个体, 全部重新分配 x.
3. 再求平均值 y_mean, 小于平均值的个体, 重新分配 x.
4. 2, 3, 步骤迭代 n 次, 取当前所有个体中的最大值 y_max.

"""

import matplotlib.pyplot as plt
import math
import random


def plot_obj_func():
    x1 = [i / float(10) for i in range(0, 100, 1)]
    y1 = [10 * math.sin(5 * x) + 7 * math.cos(4 * x) for x in x1]
    plt.plot(x1, y1)
    plt.show()
    return


def plot_current_individual(x, y):
    x1 = [i / float(10) for i in range(0, 100, 1)]
    y1 = [10 * math.sin(5 * x) + 7 * math.cos(4 * x) for x in x1]
    plt.plot(x1, y1)
    plt.scatter(x, y, c='r', s=5)
    plt.show()
    return


def plot_iter_curve(iter, results):
    x = [i for i in range(iter)]
    y = [results[i][1] for i in range(iter)]
    plt.plot(x, y)
    plt.show()
    return


def binary2decimal(binary, upper_limit, chromosome_length):
    t = 0
    for j in range(len(binary)):
        t += binary[j] * 2 ** j
    t = t * upper_limit / (2 ** chromosome_length - 1)
    return t


def init_population(pop_size, chromosome_length):
    pop = [[random.randint(0, 1) for i in range(chromosome_length)] for j in range(pop_size)]
    return pop


def decode_chromosome(pop, chromosome_length, upper_limit):
    x = []
    for elem in pop:
        temp = 0
        for i, coff in enumerate(elem):
            temp += coff * (2 ** i)
        x.append(temp * upper_limit / (2 ** chromosome_length - 1))
    return x


def calc_obj_value(pop, chromosome_length, upper_limit):
    obj_value = []
    x = decode_chromosome(pop, chromosome_length, upper_limit)
    for elem in x:
        obj_value.append(10 * math.sin(5 * elem) + 7 * math.cos(4 * elem))
    return obj_value


def calc_fit_value(obj_value):
    fit_value = []
    c_min = 10
    for value in obj_value:
        if value > c_min:
            temp = value
        else:
            temp = 0.0
        fit_value.append(temp)
    return fit_value


def find_best(pop, fit_value):
    best_individual = []
    best_fit = fit_value[0]
    for i in range(1, len(pop)):
        if fit_value[i] > best_fit:
            best_fit = fit_value[i]
            best_individual = pop[i]
    return best_individual, best_fit


def cum_sum(fit_value):
    temp = fit_value[:]
    for i in range(len(temp)):
        fit_value[i] = sum(temp[:i + 1])
    return


def selection(pop, fit_value):
    """
    首先是计算个体适应度总和，然后在计算各自的累积适应度。这两步都好理解，
    主要是第三步，转轮盘选择法。这一步首先是生成基因总数个0-1的小数，
    然后分别和各个基因的累积个体适应度进行比较。
    如果累积个体适应度大于随机数则进行保留，否则就淘汰。
    这一块的核心思想在于：一个基因的个体适应度越高，
    他所占据的累计适应度空隙就越大，也就是说他越容易被保留下来。
    :param pop:
    :param fit_value:
    :return:
    """
    p_fit_value = []
    total_fit = sum(fit_value)
    for i in range(len(fit_value)):
        p_fit_value.append(fit_value[i] / total_fit)

    cum_sum(p_fit_value)
    pop_len = len(pop)

    ms = sorted([random.random() for i in range(pop_len)])
    fitin = 0
    newin = 0
    newpop = pop[:]

    while newin < pop_len:
        if ms[newin] < p_fit_value[fitin]:
            newpop[newin] = pop[fitin]
            newin += 1
        else:
            fitin += 1
        pop = newpop[:]
    return


def crossover(pop, pc):
    pop_len = len(pop)
    for i in range(pop_len - 1):
        if random.random() < pc:
            cpoint = random.randint(0, len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0:cpoint])
            temp1.extend(pop[i + 1][cpoint: len(pop[i])])
            temp2.extend(pop[i + 1][0: cpoint])
            temp2.extend(pop[i][cpoint: len(pop[i])])
            pop[i] = temp1[:]
            pop[i + 1] = temp2[:]
    return


def mutation(pop, pm):
    px = len(pop)
    py = len(pop[0])
    for i in range(px):
        if random.random() < pm:
            mpoint = random.randint(0, py - 1)
            if pop[i][mpoint] == 1:
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1
    return


def main():
    plot_obj_func()
    pop_size = 500
    upper_limit = 10
    chromosome_length = 10
    iter = 500
    pc = 0.6
    pm = 0.01
    results = []
    pop = init_population(pop_size, chromosome_length)
    best_x = []
    best_y = []
    for i in range(iter):
        obj_value = calc_obj_value(pop, chromosome_length, upper_limit)
        fit_value = calc_fit_value(obj_value)
        best_individual, bese_fit = find_best(pop, fit_value)
        results.append([binary2decimal(best_individual, upper_limit, chromosome_length), bese_fit])
        selection(pop, fit_value)
        crossover(pop, pc)
        mutation(pop, pm)
        if iter % 200 == 0:
            best_x.append(results[-1][0])
            best_y.append(results[-1][1])
    print("x = %f, y = %f" % (results[-1][0], results[-1][1]))
    plt.scatter(best_x, best_y, s=3, c='r')
    x1 = [i / float(10) for i in range(0, 100, 1)]
    y1 = [10 * math.sin(5 * x) + 7 * math.cos(4 * x) for x in x1]
    plt.plot(x1, y1)
    plt.show()
    plot_iter_curve(iter, results)
    return


if __name__ == '__main__':
    main()


















