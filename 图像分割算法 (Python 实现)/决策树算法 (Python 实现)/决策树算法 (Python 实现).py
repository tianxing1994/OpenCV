"""
参考链接:
https://www.cnblogs.com/sxron/p/5471078.html
该方法使用信息增益法
"""
import copy
import numpy as np
from collections import  Counter


def get_majority_class_count(y_train):
    """
    :param y_train: 是训练数据类别的 ndarray, 只有一行.
    :return: 此函数找到数量最多的类别
    """
    counter_dict = dict(Counter(y_train))
    counter_sorted = sorted(counter_dict.items(), key=lambda x: x[1], reverse=True)
    result = counter_sorted[0][0]
    return result


def calc_information_entropy(y_train):
    """计算一组值的熵,
    当该组值中, 只有一个唯一值时, 返回值为 0.
    """
    total = len(y_train)
    counter_dict = dict(Counter(y_train))
    information_entropy = 0.0
    for k, v in counter_dict.items():
        prob = v / total
        information_entropy -= prob * np.log2(prob)
    return information_entropy


def split_data(train_data, column, value):
    """将训练数据中指定列为指定值的数据取出, 并去除指定列. 以此作为指定列(特征)分类后的决策树分枝."""
    target_row = train_data[np.where(train_data[:, column]==value)]
    result = np.delete(target_row, column, axis=1)
    return result


def calc_classify_entropy(train_data, column):
    """计算对数据集给定列分类后, 分类结果的熵, 和所需要消灭的熵"""
    # 该值为: 对指定列分类所需要消灭的熵.
    classify_entropy = calc_information_entropy(train_data[:, column])
    total = len(train_data)
    counter_dict = dict(Counter(train_data[:, column]))
    after_classify_entropy = 0.0
    for k, v in counter_dict.items():
        # 在对 column 指定列分类时, 当前值 k 出现的概率.
        prob = counter_dict[k] / total
        # 当 column 列为 k 值时, 求子数据集的熵
        sub_data = split_data(train_data, column, v)
        information_entropy = calc_information_entropy(sub_data[:, -1])
        # 累加熵 (熵的计算公式: -p*log(p))
        after_classify_entropy += - prob * information_entropy
    return after_classify_entropy, classify_entropy


def find_best_feature(train_data):
    feature_num = train_data.shape[1] - 1
    # 计算当前数据集分类结果的熵
    base_entropy = calc_information_entropy(train_data[:, -1])
    best_information_gain = 0.0
    best_feature = -1
    for i in range(feature_num):
        after_classify_entropy, classify_entropy = calc_classify_entropy(train_data, i)
        if classify_entropy == 0:
            # 当一个特征中只有一个值时, 其分类所需消灭的熵为 0, 信息增益也会为 0.
            # 如果这里 continue 跳过, 则当前情况下, 永远不会再对该特征进行分类, 但在对测试数据分类时, 该特征可能并不是只有一个值.
            # 所以我们这里优先对该特征进行分类.
            best_information_gain, best_feature = float('inf'), i
            break
        information_gain = (base_entropy - after_classify_entropy) / classify_entropy
        if information_gain > best_information_gain:
            best_information_gain, best_feature = information_gain, i
    return best_information_gain, best_feature


def create_classify_tree(train_data):
    if len(np.unique(train_data[:, -1])) == 1:
        return train_data[0, -1]
    if train_data.shape[1]==1:
        return get_majority_class_count(train_data[:, -1])

    _, best_feature = find_best_feature(train_data)
    the_tree = {best_feature: {}}

    values = np.unique(train_data[:, best_feature])

    for value in values:
        the_tree[best_feature][value] = create_classify_tree(split_data(train_data, best_feature, value))
    return the_tree


def classify(x_test, dtrees):
    index = int(list(dtrees.keys())[0])
    second_dict = dtrees[index]
    value = x_test[index]
    available_value = list(second_dict.keys())
    if value in available_value:
        key = value
    else:
        key = available_value[0]

    if isinstance(second_dict[key], dict):
        second_test = copy.deepcopy(x_test)
        second_test = np.delete(second_test, index, axis=0)

        class_label = classify(second_test, second_dict[key])
    else:
        class_label = second_dict[key]
    return class_label


def test_func(test_data, dtrees):
    for temp in test_data:
        class_label = classify(temp, dtrees)
        print("%s: %s" % (temp, class_label))
    return




if __name__ == '__main__':
    train_data = np.array([['sunny', 'hot', 'high', 'false', 'N'],
                           ['sunny', 'hot', 'high', 'true', 'N'],
                           ['overcast', 'hot', 'high', 'false', 'Y'],
                           ['rain', 'mild', 'high', 'false', 'Y'],
                           ['rain', 'cool', 'normal', 'false', 'Y'],
                           ['sunny', 'cool', 'high', 'true', 'N'],
                           ['overcast', 'hot', 'normal', 'true', 'Y'],
                           ['rain', 'mild', 'high', 'false', 'Y'],
                           ['rain', 'cool', 'normal', 'false', 'Y'],
                           ['rain', 'cool', 'normal', 'true', 'N'],
                           ['overcast', 'cool', 'normal', 'true', 'Y']], dtype='<U8')

    test_data = np.array([['sunny', 'mild', 'high', 'false'],
                          ['sunny', 'cool', 'normal', 'false'],
                          ['rain', 'mild', 'normal', 'false'],
                          ['sunny', 'mild', 'normal', 'true'],
                          ['overcast', 'mild', 'high', 'true'],
                          ['overcast', 'hot', 'normal', 'false'],
                          ['rain', 'mild', 'high', 'true']], dtype='<U8')

    dtree = create_classify_tree(train_data)

    result = list()
    for sample in test_data:
        class_label = classify(sample, dtree)
        result.append(class_label)
    print(result)

