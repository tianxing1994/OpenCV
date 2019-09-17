# coding=utf-8

# 首先利用sklearn的库进行knn算法的建立与预测
from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()      # 调用分类器赋在变量knn上

iris = datasets.load_iris()     # 返回一个数据库，赋值在iris上


knn.fit(iris.data, iris.target) # fit的第一个参数 是特征值矩阵，第二个参数是一维的向量

predictedLabel = knn.predict([[0.1,0.2,0.3,0.4]])

print(knn.score())