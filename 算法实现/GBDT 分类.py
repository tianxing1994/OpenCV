"""
参考链接:
blog.csdn.net/bf02jgtrs00xktcx/article/details/82719765
blog.csdn.net/Luqiang_Shi/article/details/85017280
"""
from copy import copy
import random
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


# <!-------------- 以上是回归树 --------------->
class GradientBoostingClassifier(object):
    def __init__(self):
        self.trees = None
        self.lr = None
        self.init_val = None
        self.threshold = 0.5

    @staticmethod
    def sigmoid(x):
        return 1 / (1+np.exp(-x))

    @staticmethod
    def _get_init_val(y):
        """
        设有 m 个样本, 每个样本由 n 个 1/0 的值组成. 令每个样本中 1 的比率为 yi, 为 0 的比率为 1-yi.
        则整个样本集中, 1 出现的概率为 p=(1/m) * ∑yi
        由 sigmoid 函数: p = 1 / (1+e^(-z))
        则 z 的期望:
        1/p - 1 = e^(-z)
        p / (1-p) = e^z
        z = log(p / (1-p)) = log(∑yi / ∑(1-yi)) = log(sum(yi) / sum(1-yi))
        :param y:
        :return:
        """
        m = len(y)
        y_sum = sum(y)
        return np.log(y_sum / (m - y_sum))

    @staticmethod
    def _match_node(row, tree):
        """给定一条样本, 匹配出其在回归树中的哪一个节点."""
        nd = tree.root
        while nd.left and nd.right:
            if row[nd.feature] < nd.split:
                nd = nd.left
            else:
                nd = nd.right
        return nd

    @staticmethod
    def _get_leaves(tree):
        """获取所有的叶子节点. 即不同时包含 node.left, node.right 的节点."""
        nodes = list()
        que = [tree.root]
        while len(que) != 0:
            node = que.pop(0)
            if node.left is None or node.right is None:
                nodes.append(node)
                continue
            left_node = node.left
            right_node = node.right
            que.append(left_node)
            que.append(right_node)
        return nodes

    def _divide_regions(self, tree, nodes, X):
        """将样本集 X 中的每一条样本分配到其所对应的 node 叶子节点中. """
        regions = {node: list() for node in nodes}
        for i, row in enumerate(X):
            node = self._match_node(row, tree)
            regions[node].append(i)
        return regions

    def _get_score(self, indexs, y_hat, residuals):
        """
        残差函数: fm(X) = ∑(yi-pi) / ∑(pi*(1-pi))
        其中
        yi 代表对应样本的真实类别为 1 的概率.
        pi 代表对应样本的预测类别为 1 的概率.
        :param indexs:
        :param y_hat: 由回归树返回的值.
        :param residuals: 真实值减去回归树的预测值经过 sigmoid 计算后得到的概率.
        :return:
        """
        numerator = sum(residuals[indexs])
        denominator = sum(self.sigmoid(y_hat[indexs]) * (1 - self.sigmoid(y_hat[indexs])))
        return numerator / denominator

    def _update_score(self, tree, X, y_hat, residuals):
        """
        博主没有采用预测残差的回归树的值, 而采用了另一种计算方法.
        这里需要将已生成的回归树中的值进行更改.
        :param tree:
        :param X:
        :param y_hat: 由回归树返回的值.
        :param residuals: 由 y_hat 经过 sigmoid 计算后得到的概率与真实类别 y 之间的差值.
        :return:
        """
        nodes = self._get_leaves(tree)
        regions = self._divide_regions(tree, nodes, X)
        for node, indexs in regions.items():
            node.score = self._get_score(indexs, y_hat, residuals)
        tree._get_rules()

    def fit(self, X, y, n_estimators, lr, max_depth, min_samples_split, subsample=None):
        self.init_val = self._get_init_val(y)
        m = len(y)
        y_hat = np.array([self.init_val] * m)

        residuals = y - self.sigmoid(y_hat)

        self.trees = list()
        self.lr = lr
        for _ in range(n_estimators):
            index = range(m)
            if subsample is not None:
                k = int(subsample * m)
                index = random.choices(population=index, k=k)
            X_sub = X[index]
            residuals_sub = residuals[index]
            y_hat_sub = y_hat[index]

            # 拟合回归树, 用于预测残差.
            tree = RegressionTree()
            tree.fit(X_sub, residuals_sub, max_depth, min_samples_split)
            # 看起来已经拟合了的回归树就是用来预测残差的, 但是博主采用了一种方法, 计算残差.
            # 这里需要将叶子节点的值进行更改.
            self._update_score(tree, X_sub, y_hat_sub, residuals_sub)

            # 按学习率, 将残差回归树预测的残差应用到 y_hat 中.
            y_hat = y_hat + lr * tree.predict(X)

            # 获取新的残差.
            residuals = y - self.sigmoid(y_hat)
            # 将残差回归树添加到弱估计器列表.
            self.trees.append(tree)

    def _predict(self, X):
        return self.sigmoid(self.init_val + sum(self.lr * tree.predict(X) for tree in self.trees))

    def predict(self, X):
        return np.array(self._predict(X) >= self.threshold, dtype=np.int)


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    print("Testing the accuracy of GBDT classifier...")
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train, n_estimators=10,
            lr=0.8, max_depth=10, min_samples_split=2)

    result = clf.predict(X)
    print(result)
    print(y)
