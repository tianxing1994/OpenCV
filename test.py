from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


data = ['TF-IDF 算法 的 主要 思想 是',
        '算法 一个 重要 特点 可以 脱离 语料库 背景',
        '如果 一个 网页 被 很多 其他 网页 链接 说明 网页 重要',
        'TF-IDF 算法 的 原理 很 简单',
        'TF-IDF 算法 的 应用 很 广泛']

vectorizer = CountVectorizer(max_features=5)
vocabulary_matrix = vectorizer.fit_transform(data)
print(vectorizer.vocabulary_)
print(vocabulary_matrix)

tf_idf_transformer = TfidfTransformer(norm=None)
tf_idf = tf_idf_transformer.fit_transform(vocabulary_matrix)
x_train_weight = tf_idf.toarray()
print(x_train_weight)
