"""
参考链接:
https://www.cnblogs.com/ahu-lichang/p/7157855.html

贝叶斯公式: P(A|B)=P(B|A)*P(A)/P(B)
"""
import numpy as np


class TfidfVectorizerMine(object):
    def __init__(self):
        self._bag_of_words = None

    def _build_bag_of_words(self, documents):
        bag = set()
        for document in documents:
            bag = bag | set(document)
        return list(bag)

    def fit(self, documents):
        """建立 BOW 词袋"""
        self._bag_of_words = self._build_bag_of_words(documents)
        return self._bag_of_words

    def _document2vector(self, document):
        vector = [0] * len(self._bag_of_words)
        for word in document:
            if word in self._bag_of_words:
                vector[self._bag_of_words.index(word)] += 1
        return vector

    def _documents2vector(self, documents):
        vecters = []
        for document in documents:
            vector = self._document2vector(document)
            vecters.append(vector)
        return np.array(vecters)

    def transform(self, documents):
        """根据当前的 BOW 词袋, 将文档集转换为向量."""
        if self._bag_of_words is None:
            raise ValueError("please fit documents first.")
        result = self._documents2vector(documents)
        return result

    def fit_transform(self, documents):
        """该方法只是简单地将 fit 和 transform 方法结合起来. """
        self.fit(documents)
        result = self.transform(documents)
        return result


class NaiveBayes(object):
    def __init__(self):
        self._pkia = None
        self._pkib = None
        self._pa = None
        self._pb = None

    def _calc_priori_probability(self, x_train, y_train):
        """
        根据训练样本, 计算 P(ki|A), P(ki|B), P(A), P(B).
        :return:
        """
        num_of_samples, num_of_features = x_train.shape[:2]
        total_a = np.sum(y_train)
        pa = total_a / num_of_samples
        pb = 1.0 - pa
        pa_sum = np.ones(shape=num_of_features)
        pb_sum = np.ones(shape=num_of_features)
        pa_total_eig, pb_total_eig = 2.0, 2.0
        for i in range(num_of_samples):
            if y_train[i] == 1:
                pa_sum += x_train[i]
                pa_total_eig += np.sum(x_train[i])
            else:
                pb_sum += x_train[i]
                pb_total_eig += np.sum(x_train[i])
        pkia = np.log(pa_sum / pa_total_eig)
        pkib = np.log(pb_sum / pb_total_eig)
        return pkia, pkib, pa, pb

    def fit(self, x_train, y_train):
        """根据训练样本, 统计先验概率."""
        pkia, pkib, pa, pb = self._calc_priori_probability(x_train, y_train)
        self._pkia, self._pkib, self._pa, self._pb = pkia, pkib, pa, pb
        return pkia, pkib, pa, pb

    def _calc_probability(self, x_predict):
        pa = np.sum(x_predict * self._pkia, axis=1) + np.log(self._pa)
        pb = np.sum(x_predict * self._pkib, axis=1) + np.log(self._pb)
        result = np.where(pa > pb, 1, 0)
        return result

    def predict(self, x_predict):
        result = self._calc_probability(x_predict)
        return result


if __name__ == '__main__':
    documents = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    label = [0, 1, 0, 1, 0, 1]

    test_document = [['love', 'my', 'dalmation'],
                     ['stupid', 'garbage']]

    tf = TfidfVectorizerMine()

    x_train = tf.fit_transform(documents)
    y_train = label

    x_pred = tf.transform(test_document)

    mNB = NaiveBayes()
    mNB.fit(x_train, y_train)
    result = mNB.predict(x_pred)
    print(result)
