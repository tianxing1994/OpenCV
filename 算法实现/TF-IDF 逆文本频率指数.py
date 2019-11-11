"""
参考链接:
https://blog.csdn.net/asialee_bird/article/details/81486700
https://blog.csdn.net/xieyan0811/article/details/89790729
http://www.ruanyifeng.com/blog/2013/03/tf-idf.html

TF-IDF 是一种用于信息检索与文本挖掘的常用加权技术.

TF-IDF 是一种统计方法, 用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度.
字词的重要性随着它在文件中出现的次数成正比增加, 但同时会随着它在语料库中出现的频率成反比下降.
TF-IDF 的主要思想是: 如果某个单词在一篇文章中出现的频率 TF 高, 并且在其他文章中很少出现,
则认为此词或者短语具有很好的类别区分能力, 适合用来分类.

1. TF 是词频 (Term Frequency)
词频 (TF) 表示词条 (关键字) 在文本中出现的频率.
这个数字通常会被归一化(一般是词频除以文章总词数), 以免它偏向长的文件.

TF_{ij} = n_{ij} / ∑n_{ij}
即: TF_{ω} = "在某一类中词条 ω 出现的次数" / "该类中所有词条数目"

其中 n_{ij} 是该词在文件 dj 中出现的次数, 分母则是文件 dj 中所有词汇出现的次数总和.

2. IDF 逆向文本频率 (Inverse Document Frequency)
逆向文件频率(IDF): 某一特定词语的 IDF, 可以由总文件数目除以包含该词语的文件数目, 再将得到的商取对数得到.
如果包含词条 t 的文档越少, IDF 越大, 则说明词条具有很好的类别区分能力.

IDF_{i} - log(|D| / |{j:t_{i} ∈ d_{j}}|)

其中, |D| 是语料库中的文件总数. |{j:t_{i} ∈ d_{j}}| 表示包含词语 t_{i} 的文件数目 (即 n_{i}, n_{j} 的文件数目).
如果该词语不在语料库中, 就会导致分母为零, 因此一般情况下使用 1 + |{j:t_{i} ∈ d_{j}}|

IDF = log("语料库的文档总数" / ("包含该词的文档数" + 1))
分母加 1, 以免分母为 0, 0 不在 log 对数函数的定义域.

3. TF-IDF 实际上是: TF * IDF
某一特定文件内的高频率词语, 以及该词语在整个文件集合中的为低频率词语时, 可以产生出高权重的 TF-IDF.
因此, TF-IDF 你若倾向于过滤掉常见的词语, 保留重要的词语.

TF-IDF = TF * IDF

注: TF-IDF 算法非常容易理解, 并且很容易实现, 但是其简单结构并没有考虑词语的语义信息, 无法处理一词多义与一义多词的情况.

TF-IDF 常用于: 搜索引擎, 关键词提取, 文本相似性, 文本摘要
"""
from collections import defaultdict
import numpy as np


def load_dataset():
    data = np.array([['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                     ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                     ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                     ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                     ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                     ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']])

    target = np.array([0, 1, 0, 1, 0, 1])
    return data, target


def feature_select(data):
    # 统计整个数据集中各词的数量.
    total_words_count = defaultdict(int)
    for doc in data:
        for word in doc:
            total_words_count[word] += 1

    # 计算整个数据集中各词的频率 TF.
    words_tf = dict()
    for word in total_words_count.keys():
        words_tf[word] = total_words_count[word] / sum(total_words_count.values())

    # 计算整个数据集中各词的逆向词频 IDF.
    doc_num = len(data)
    # 存储各词的 IDF 值.
    words_idf = dict()
    # 存储包含各词的文档数.
    words_doc = defaultdict(int)
    for word in total_words_count.keys():
        for sample in data:
            if word in sample:
                words_doc[word] += 1

    for word in total_words_count.keys():
        words_idf[word] = np.log(doc_num / (words_doc[word] + 1))

    words_tfidf = dict()
    for word in total_words_count.keys():
        words_tfidf[word] = words_tf[word] * words_idf[word]

    # 对字典按值由大到小排序.
    dict_feature_select = sorted(words_idf.items(), key=lambda x: x[1], reverse=True)
    return dict_feature_select


def demo1():
    data, target = load_dataset()
    features = feature_select(data)

    print(features)
    print(len(features))
    return


def demo2():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    x_train = ['TF-IDF 主要 思想 是', '算法 一个 重要 特点 可以 脱离 语料库 背景',
               '如果 一个 网页 被 很多 其他 网页 链接 说明 网页 重要']
    x_test = ['原始 文本 进行 标记', '主要 思想']

    # 该类会将文本中的词语转换为词频矩阵, 矩阵元素a[i][j] 表示 j 词在 i 类文本下的词频
    vectorizer = CountVectorizer(max_features=10)
    # 该类会统计每个词语的tf-idf权值
    tf_idf_transformer = TfidfTransformer()
    # 将文本转为词频矩阵并计算tf-idf
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(x_train))
    # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    x_train_weight = tf_idf.toarray()

    # 对测试集进行tf-idf权重计算
    tf_idf = tf_idf_transformer.transform(vectorizer.transform(x_test))
    x_test_weight = tf_idf.toarray()  # 测试集TF-IDF权重矩阵

    print('输出x_train文本向量：')
    print(x_train_weight)
    print('输出x_test文本向量：')
    print(x_test_weight)
    return


if __name__ == '__main__':
    demo2()
































