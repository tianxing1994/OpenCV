# coding=utf-8
"""
参考链接:
https://blog.csdn.net/asialee_bird/article/details/81051281
"""
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def image2vector(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
    string = data.replace('\n', '')
    l = list(string)
    result = np.array(l, dtype=np.int)
    return result


def load_data(dirname):
    training_file_list = os.listdir(dirname)
    m = len(training_file_list)

    data = np.zeros(shape=(m, 1024), dtype=np.int)
    target = np.zeros(shape=(m,), dtype=np.int)

    for i in range(m):
        filename = training_file_list[i]
        classify = int(filename.split('_')[0])
        data[i] = image2vector(os.path.join(dirname, filename))
        target[i] = classify
    return data, target


def knn_demo():
    data, target = load_data('../dataset/nlp/data_digits')
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    knn = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    score = knn.score(X_test, y_test)

    print(y_pred)
    print(score)
    return


if __name__ == '__main__':
    knn_demo()
