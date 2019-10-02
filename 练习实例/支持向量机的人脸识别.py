"""
参考链接
https://blog.csdn.net/qq_41302130/article/details/82937698

理解:
1. 他这个人脸识别是比较简单, 就是 SVC 分类.
2. 我本来想把他的 PCA 降维改成 LDA 降维, 但是不知道为什么 LDA 降维要求 n_component 小于 n_classes.
3. 另外还发现, 我自己实现的那个 LDA 降维会产生虚数部分.
"""
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, recall_score, accuracy_score
from sklearn.metrics import classification_report

import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    # flw 数据集中有很多人名对应其照片, 每个人的照片数量不定, min_faces_per_person=60 是指提取有 60 张照片以上的人的数据.
    # 查看 fetch_lfw_people 的 doc 参数信息, 有更多关于数据的提示.
    faces = fetch_lfw_people(min_faces_per_person=60)

    # print(faces)
    # print(faces.images.shape)
    # print(faces.data)
    # print(faces.target)
    # print(faces.target_names)

    X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.2)

    pca = PCA(n_components=150, whiten=True, random_state=42)
    svc = SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(pca, svc)

    param_grid = {'svc__C': [1, 5, 10], 'svc__gamma': [0.0001, 0.0005, 0.001]}  # 选取要循环的参数，就是要测试的参数
    grid = GridSearchCV(model, param_grid=param_grid, cv=5, iid=True)  # 选择模型，选择CV，就是交叉验证，如果不进行结果不准确
    grid.fit(X_train, y_train)

    # print(grid.best_params_)
    # print(grid.best_score_)

    # CV验证可以把最优的模型直接拿出来
    model = grid.best_estimator_
    # 拿到最好的模型后，进行预测
    y_pred = model.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    # 把小数位数改为2位
    np.set_printoptions(precision=2)
    print(accuracy_score(y_test, y_pred))
    print(cnf_matrix)

    # 导入分类报告
    print(classification_report(y_test, y_pred, target_names=faces.target_names))
    return


if __name__ == '__main__':
    demo1()


