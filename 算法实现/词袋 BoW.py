"""
这是一个简单的词向量特征提取方法, 虽然我不知道, 也没有参考 TfidfVectorizer 的实现方法.
当此处实现的方法, 可以大质实现 TfidfVectorizer 的功能. 即: 将文本的样本转换为可训练的数值样本.

BOW: bag of words, 词袋, 是一种特征提取方法. 这里, 我们将每一个文档当作是一个对象. 文档中的每一个词, 当作是一个特征.
则: 同一类的文档, 其相应特征出现的频率应该是具有相关性的. 因此, 这样的特征提取, 可以用于文档分类.
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
