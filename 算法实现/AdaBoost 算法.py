"""
https://blog.csdn.net/weixin_43383558/article/details/84860078

AdaBoost 算法
步骤:
1. 选择一个估计器作为元估计器.
2. 训练元估计器并预测训练集. 保存训练好的第1个估计器及预测结果得分.
3. 将第1个估计器没有正确预测的样本的权重加大. 训练第2个估计器并预测训练集. 保存训练好的第2个估计器及预测结果得分.
4. 以上步骤重复, 直到 n 个估计器. 或其它停止条件.
5. 根据每个估计器的预测结果得分, 给每个估计器划分权重.
6. 使用每个估计器的预测结果, 对最终结果进行带权重的投票. 得出最终分类结果


算法理解:
设想有一个线性回归分类器. 此时它只能用一条直线去分割样本.
显然, 线性回归分类器. 很有可能不能百分百地将样本分开 (因为样本非线性可分, 需要曲线才能分开).
此时, 如果我们调整每个样本的权重. 并拟合(训练)线性回归分类器. 可想而知, 此次的直线和之前的直线将不一样.
即然, 通过改变样本的权重, 可以训练出不同分割线的线性回归分类器.
那么, 我们将这些线性分类器按权重合到一起, 此时便可得到一条曲线.
此曲线可以将原样本完全分开.
AdaBoost 算法, 则是在寻找这样一条曲线.

"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams['font.family'] = 'SimHei'   # 用来正确显示中文
plt.rcParams['axes.unicode_minus'] = False      # 用来正确显示负号


def split_dataset(dataset, index, value, types='le'):
    """
    根据指定列 i, 对 dataset 数据集进行分类.
    :param dataset: ndarray, 形状为 (m, n), 其中最后一列代表类别, 样本 x (m, n-1) 类别 y (m, 1)
    :param index: int, 特征的下标索引.
    :param value: float. 阈值
    :param types: str, 取值 'le' 或 'gt'.
    types='le' 时, 当 i 列的值小于等于 value 时, 该类为 -1;
    types='gt' 时, 当 i 列的值大于 value 时, 该类为 -1.
    :return: ndarray, 形状为 (m, 1), 值为 1 或 -1. 表示对 dataset 数据集的分类.
    """
    m, _ = dataset.shape
    ret_array = np.ones(shape=(m, 1))
    if types == 'le':
        ret_array[dataset[:, index] <= value] = -1
    elif types == 'gt':
        ret_array[dataset[:, index] > value] = -1
    return ret_array


def build_simple_tree(dataset, y, d):
    """
    从 dataset 中选择一个特征 index, 在这个特征中选择一个分割点 value, 及分割类型 types, 对样本进行分类.
    找出分类正确率最高的特征 index, 分割点, value, 分割类型 types.
    :param dataset: ndarray, 形状为 (m, n), 其中最后一列代表类别, 样本 x (m, n-1) 类别 y (m, 1)
    :param y: ndarray, 形状为 (m, 1), 代表 dataset 中每条样本的类别. y 跟 dataset 中的最后一列的值一样.
    :param d: ndarray, 形状为 (m, 1), 代表 dataset 中每条样本的权重.
    :return: best_tree, min_error, best_class_estimated.
    best_tree, 根据给定的样本权重 d, 得出的最佳分类特征/分割点/分割类型的字典;
    min_error, 考虑样本权重 d 后, best_tree 的分割正确率. float
    best_class_estimated, best_tree 的分类结果. ndarray 形状为 (m, 1)
    例如:
    best_tree={'index': 0, 'value': 1.3, 'types': 'le'};
    min_error=0.2;
    best_class_estimated=np.array([[-1.], [ 1.], [-1.], [-1.], [ 1.]])
    """
    m, n = dataset.shape
    num_features = n - 1
    num_steps = 10
    min_error = np.inf
    best_tree = dict()
    for index in range(num_features):
        range_min = dataset[:, index].min()
        range_max = dataset[:, index].max()
        step_size = (range_max - range_min) / num_steps
        for j in range(-1, int(num_steps) + 1):
            for types in ['le', 'gt']:
                value = (range_min + float(j) * step_size)
                predict_values = split_dataset(dataset, index, value, types)
                error_array = np.ones(shape=(m, 1))
                error_array[predict_values == y] = 0
                weighted_error = np.dot(d.T, error_array)
                weighted_error = np.squeeze(weighted_error)
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_estimated = predict_values.copy()
                    best_tree['index'] = index
                    best_tree['value'] = value
                    best_tree['types'] = types
    return best_tree, min_error, best_class_estimated


def adaboost(dataset, max_loop=100):
    """
    基于单层决策树的 adaboost 分类器
    :param dataset: ndarray, 形状为 (m, n), 其中最后一列代表类别, 样本 x (m, n-1) 类别 y (m, 1)
    :param max_loop: int, 最大迭代次数
    :return: 一系列弱分类器及其权重, 样本分类结果
    """
    adaboost_tree = list()
    m, n = dataset.shape
    y = dataset[:, -1].reshape((-1, 1))

    # 初始化权重, 1/m
    d = np.array(np.ones(shape=(m, 1)) / m)
    agg_class_estimated = np.mat(np.zeros(shape=(m, 1)))
    for i in range(max_loop):
        best_tree, min_error, best_class_estimated = build_simple_tree(dataset, y, d)
        # alpha=0.5*log(正确率/错误率). 作为其对预测结果投票的权重.
        alpha = 0.5*np.log((1-min_error) / (min_error + 1e-5))
        best_tree['alpha'] = alpha
        adaboost_tree.append(best_tree)
        # y * best_class_estimated. 返回形状为 (m, 1) 的 ndarray. 预测正确的为 1, 错误的为 -1.
        # -alpha * y * best_class_estimated. 预测正确的为负数, 错误的为正数.
        # np.exp() 正数返回值大于 1, 负数返回值小于 1.
        # 预测正确的为负数, 权重 d 乘以一个小于 1 的数, 权重减小; 预测错误的为正数, 权重 d 乘以一个大于 1 的数, 权重增大.
        d *= np.exp(-alpha * y * best_class_estimated)
        # 将新的权重的和转换为 1.
        d /= d.sum()
        # 对最终结果投票.
        agg_class_estimated += alpha * best_class_estimated
        agg_errors = np.multiply(np.sign(agg_class_estimated) != np.mat(y), np.ones(shape=(m, 1)))
        error_rate = agg_errors.sum() / m
        # 当前的错误率.
        print("error rate: ", error_rate)
        if error_rate == 0.0:
            break
    return adaboost_tree, agg_class_estimated


def adaclassify(data, adaboost_tree):
    """
    对预测数据进行分类
    :param data: 预测样本 x 及 y
    :param adaboost_tree: 使用训练数据, 训练好的决策树
    :return: 预测样本分类结果
    """
    data_matrix = np.mat(data)
    m = np.shape(data_matrix)[0]
    agg_class_estimated = np.mat(np.zeros(shape=(m, 1)))
    for i in range(len(adaboost_tree)):
        class_estimated = split_dataset(data_matrix,
                                        adaboost_tree[i]['index'],
                                        adaboost_tree[i]['value'],
                                        adaboost_tree[i]['types'])
        agg_class_estimated += adaboost_tree[i]['alpha'] * class_estimated
    result = np.sign(agg_class_estimated)
    return result


def plot_data(dataset):
    """数据画图"""
    # 取两类 x1 及 x2 值画图
    type1_x1 = dataset[dataset[:, -1] == -1][:, :-1][:, 0].tolist()
    type1_x2 = dataset[dataset[:, -1] == -1][:, :-1][:, 1].tolist()
    type2_x1 = dataset[dataset[:, -1] == 1][:, :-1][:, 0].tolist()
    type2_x2 = dataset[dataset[:, -1] == 1][:, :-1][:, 1].tolist()

    # 画点
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(type1_x1, type1_x2, marker='s', s=90)
    ax.scatter(type2_x1, type2_x2, marker='o', s=50, c='red')
    plt.title('AdaBoost 训练数据')
    plt.show()
    return


if __name__ == '__main__':
    print("1, Adaboost, 开始")
    dataset = np.array([[1, 2.1, 1],
                        [2, 1.1, 1],
                        [1.3, 1, -1],
                        [1, 1, -1],
                        [2, 1, 1]])

    # print("\n2, AdaBoost 数据画图")
    # plot_data(dataset)

    print("3, 计算 AdaBoost 树")
    adaboost_tree, agg_class_estimated = adaboost(dataset)

    # 对数据进行分类
    print("4, 对 [5, 5], [0, 0] 点, 使用 AdaBoost 进行分类: ")
    print(adaclassify([[5, 5], [0, 0]], adaboost_tree))
















