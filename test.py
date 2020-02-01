# 此代码适用opencv3版本以上
from numpy import *  # 导入numpy的库函数
import cv2
import numpy as np
import matplotlib.pyplot as plt

labels = (1, -1, 1, 1, -1, -1, -1, 1, -1, -1)
labels = np.array(labels)
trainingData = np.array([[10, 3], [5, 0.5], [10, 5], [0.5, 10], [0.5, 1.6], [3, 6], [1.2, 4], [6, 6], [0.9, 5], [4, 4]],
                        dtype='float32');  # 数据集一定要设置成浮点型
# labels转换成10行1列的矩阵
labels = labels.reshape(10, 1)
# trainingData转换成10行2列的矩阵
trainingData = trainingData.reshape(10, 2)

# 创建分类器
svm = cv2.ml.SVM_create()
# 设置svm类型
svm.setType(cv2.ml.SVM_C_SVC)
# 核函数
svm.setKernel(cv2.ml.SVM_LINEAR)
# 训练
ret = svm.train(trainingData, cv2.ml.ROW_SAMPLE, labels)

# 测试数据
# 取0-10之间的整数值
arrayTest = np.empty(shape=[0, 2], dtype='float32')
for i in range(10):
    for j in range(10):
        arrayTest = np.append(arrayTest, [[i, j]], axis=0)
pt = np.array(np.random.rand(50, 2) * 10, dtype='float32')  # np.random.rand(50,2) * 10可以替换成arrayTest
# 预测
(ret, res) = svm.predict(pt)

# 按label进行分类显示
plt.figure("分类")
res = np.hstack((res))  # 在水平方向上平铺

# 第一类
type_data = pt[res == -1]
# 绘制散点图
plt.scatter(np.array(type_data[:, 0]), np.array(type_data[:, 1]), c='r', marker='o')

# 第二类
type_data = pt[res == 1]
plt.scatter(np.array(type_data[:, 0]), np.array(type_data[:, 1]), c='g', marker='o')

plt.show()


