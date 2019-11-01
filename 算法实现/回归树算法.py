"""
参考链接:
https://blog.csdn.net/iodjSVf8U1J7KYc/article/details/81611588
"""
from copy import copy
import numpy as np


class Node(object):
    """存储预测值, 左右结点, 特征和分割点"""
    def __init__(self, score=None):
        self.score = round(score, 2) if score is not None else None
        self.left = None
        self.right = None
        self.feature = None
        self.split = None


class RegressionTree(object):
    """存储根节点和树的高度"""
    def __init__(self):
        self.root = Node()
        self.height = 0

    @staticmethod
    def _get_split_mse(X, y, index, feature, split):
        """
        计算从 split 点分割样本集, 分割点两边的值的方差之和 MSE.
        注: D(X) = E[X - E[X]]^2 = E[X^2] - [E(X)]^2
        :param X: 样本集, 形状为 (m, n).
        :param y: 对应本的值(回归问题), 形状为 (n,)
        :param index: 索引, 包含整数的列表. 用于指定从 X, y 中索引样本及其类别做为子样本集. len(index) <= m
        :param feature: 索引, 整数. 用于指定 X 中的一个特征.
        :param split: 分割点, 子样本集中, feature 特征的一个值.
        :return:
        """
        sub_X = X[index]
        sub_y = y[index]
        y1 = sub_y[sub_X[:, feature] < split]
        y2 = sub_y[sub_X[:, feature] >= split]
        d_y1 = np.mean(y1**2) - np.mean(y1)**2
        d_y2 = np.mean(y2**2) - np.mean(y2)**2
        mse = d_y1 + d_y2
        return mse, split, (np.mean(y1), np.mean(y2))

    def _choose_split_point(self, X, y, index, feature):
        """
        指定特征 feature, 在子样本集中找出最佳分割点.
        :param X: 样本集, 形状为 (m, n).
        :param y: 对应本的值(回归问题), 形状为 (n,)
        :param index: 索引, 包含整数的列表. 用于指定从 X, y 中索引样本及其类别做为子样本集. len(index) <= m
        :param feature: 索引, 整数. 用于指定 X 中的一个特征.
        :return:
        """
        unique = np.unique(X[index][:, feature])
        if len(unique) == 1:
            return None
        min_mse = float('inf')
        ret_split = None
        ret_split_avg = None
        for split in unique[1:]:
            mse, split, split_avg = self._get_split_mse(X, y, index, feature, split)
            if mse < min_mse:
                min_mse = mse
                ret_split = split
                ret_split_avg = split_avg
        return min_mse, feature, ret_split, ret_split_avg

    def _choose_feature(self, X, y, index):
        """
        遍历所有特征, 计算最佳分割点对应的 MSE, 找出 MSE 最小的特征,
        对应的分割点, 左右子节点对应的均值和行号.
        如果所有的特征都没有不重复元素则返回 None.
        :param X: 样本集, 形状为 (m, n).
        :param y: 对应本的值(回归问题), 形状为 (n,)
        :param index: 索引, 包含整数的列表. 用于指定从 X, y 中索引样本及其类别做为子样本集. len(index) <= m
        :return:
        """
        n = X.shape[1]
        split_rets = [x for x in map(lambda x: self._choose_split_point(X, y, index, x), range(n)) if x is not None]

        if len(split_rets) == 0:
            return None

        _, feature, split, split_avg = min(split_rets, key=lambda x: x[0])
        index_split = [[], []]
        while index:
            i = index.pop()
            xi = X[i][feature]
            if xi < split:
                index_split[0].append(i)
            else:
                index_split[1].append(i)
        return feature, split, split_avg, index_split

    @staticmethod
    def _expr2literal(expr):
        """将规则用文字表达出来, 方便查看. """
        feature, op, split = expr
        op = ">=" if op == 1 else "<"
        return f"Feature {feature} {op} {split}"

    def _get_rules(self):
        """将回归树的所有规则用文字表达出来, 方便我们了解树的全貌. 这里用到了队列+广度优先搜索. """
        que = [(self.root, list())]
        self.rules = list()
        while len(que) != 0:
            nd, exprs = que.pop()
            if not (nd.left or nd.right):
                literals = list(map(self._expr2literal, exprs))
                self.rules.append([literals, nd.score])
            if nd.left:
                rule_left = copy(exprs)
                rule_left.append([nd.feature, -1, nd.split])
                que.append((nd.left, rule_left))
            if nd.right:
                rule_right = copy(exprs)
                rule_right.append([nd.feature, 1, nd.split])
                que.append((nd.right, rule_right))

    def fit(self, X, y, max_depth=5, min_samples_split=2):
        self.root = Node()
        que = [(0, self.root, list(range(len(y))))]
        while que:
            depth, nd, index = que.pop()
            if depth == max_depth:
                break
            if len(index) < min_samples_split or len(set(map(lambda i: y[i], index))) == 1:
                continue
            feature_rets = self._choose_feature(X, y, index)
            if feature_rets is None:
                continue
            nd.feature, nd.split, split_avg, index_split = feature_rets
            nd.left = Node(split_avg[0])
            nd.right = Node(split_avg[1])
            que.append((depth+1, nd.left, index_split[0]))
            que.append((depth+1, nd.right, index_split[1]))

            self.height = depth
            self._get_rules()

    def print_rules(self):
        for i, rule in enumerate(self.rules):
            literals, score = rule
            print(f"Rule {i}", '|'.join(literals) + '=> split_hat %.4f' % score)

    def _predict(self, row):
        nd = self.root
        while nd.left and nd.right:
            if row[nd.feature] < nd.split:
                nd = nd.left
            else:
                nd = nd.right
        return nd.score

    def predict(self, X):
        return np.array([self._predict(Xi) for Xi in X])


if __name__ == '__main__':
    print("testing the accuracy of Regression Tree.")
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    data = load_boston()
    X = data['data']
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

    reg = RegressionTree()
    reg.fit(X=X_train, y=y_train, max_depth=20)

    reg.print_rules()
    result = reg.predict(X)
    print(result)
    print(y)
