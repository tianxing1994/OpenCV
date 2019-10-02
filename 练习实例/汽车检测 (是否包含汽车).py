"""
参考文档:
http://www.yyearth.com/index.php?aid=241
数据集的下载地址:
http://www.yyearth.com/attachment.php?id=1234

BOW模型的处理过程:
1. 采用 SIFT 特征提取. 得到每张图片的特征点 keypoints 及其描述符 descriptor.
2. 汇总所有训练样本的描述符, 并采用 KMeans 聚类算法将描述符聚类成 N 个特征(将 descriptors 聚类成 N 个).
3. 计算每一张图片的特征描述符, 并映射成 KMeans 聚类得到的 N 个描述符,
统计每张图片中的各特征分别出现的次数, 转换为该图片各类描述符的比例.
作为这张图的 BOW 特征(该 BOW 特征, 具有 N 个数字, 作为 1 行).
4. 至此, 每张图片都可以得到一个具有 N 个特征的向量 (如传统机器学习所需要的数据集).
5. 有了以数字表示的 BOW 特征, 则可以对图像进行传统机器学习的分类, 聚类等操作.

问题:
本来是要使用 'dataset/data/car_data/TrainImages' 训练集训练模型. 然后使用
'dataset/data/car_data/TestImages' 测试集进行测试.
但是可能由于训练集的图片都比较小, 测试集的图片比较大. 所以 BOWImgDescriptorExtractor 检测器在测试图片上检测不到信息.
所以只好将训练集拆分成训练集和测试集了. 测试得分为 0.85. 感觉还可以.

我看了 car_data 数据集中有 trueLocations.txt, trueLocations_Scale.txt 两个文件.
应该是用来预测图片中汽车的位置用.
"""
import os
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class CarDetector(object):
    def __init__(self, train_image_folder):
        self._detect = cv.xfeatures2d.SIFT_create(nfeatures=-1)
        self._bow_kmeans_trainer = cv.BOWKMeansTrainer(60)
        flann_params = dict(algorithm=1, trees=5)
        flann = cv.FlannBasedMatcher(flann_params, {})
        self._extract_bow = cv.BOWImgDescriptorExtractor(self._detect, flann)

        self._svc = None

        self._train_image_folder = train_image_folder
        self._vocabulary = None
        self._data = None
        self._target = None

        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None

    def _calc_vocabulary(self):
        filenames = os.listdir(self._train_image_folder)
        for filename in filenames:
            image_path = os.path.join(self._train_image_folder, filename)
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            if image is None:
                continue
            _, descriptors = self._detect.detectAndCompute(image, mask=None)
            if descriptors is None:
                continue
            self._bow_kmeans_trainer.add(descriptors)

        vocabulary = self._bow_kmeans_trainer.cluster()
        return vocabulary

    @property
    def vocabulary(self):
        if self._vocabulary is not None:
            return self._vocabulary
        self._vocabulary = self._calc_vocabulary()
        return self._vocabulary

    def _extract_features(self):
        self._extract_bow.setVocabulary(self.vocabulary)
        data = list()
        target = list()
        filenames = os.listdir(self._train_image_folder)
        for filename in filenames:
            image_path = os.path.join(self._train_image_folder, filename)
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            keypoints = self._detect.detect(image, None)
            sample_feature = self._extract_bow.compute(image, keypoints)
            if sample_feature is None:
                continue
            data.extend(sample_feature)
            label = 1 if filename.split('-')[0] == 'pos' else 0
            target.append(label)
        data = np.array(data)
        target = np.array(target, dtype=np.int)
        return data, target

    @property
    def data(self):
        if self._data is not None:
            return self._data
        self._data, self._target = self._extract_features()
        return self._data

    @property
    def target(self):
        if self._target is not None:
            return self._target
        self._data, self._target = self._extract_features()
        return self._target

    def fit_svc(self, test_size=0.0):
        if test_size == 0.0:
            self._X_train, self._y_train = self.data, self.target
        else:
            self._X_train, self._X_test, self._y_train, self._y_test = \
                train_test_split(self.data, self.target, test_size=test_size)
        svc = SVC(kernel="linear", probability=True)
        svc.fit(self._X_train, self._y_train)
        self._svc = svc
        return self._svc

    @property
    def svc(self):
        if self._svc is not None:
            return self._svc
        raise AttributeError('CarDetector object has not attribute svc.')

    @property
    def X_test(self):
        if self._X_test is not None:
            return self._X_test
        raise AttributeError('CarDetector object has not attribute X_test.')

    @property
    def y_test(self):
        if self._y_test is not None:
            return self._y_test
        raise AttributeError('CarDetector object has not attribute y_test.')

    def extract_features(self, image):
        keypoints = self._detect.detect(image, None)
        sample_feature = self._extract_bow.compute(image, keypoints)
        return sample_feature


def demo1():
    """封装成类."""
    train_image_folder = '../dataset/data/car_data/TrainImages'
    car_detector = CarDetector(train_image_folder=train_image_folder)
    car_detector.fit_svc(test_size=0.2)
    y_pred = car_detector.svc.predict(car_detector.X_test)
    y_prob = car_detector.svc.predict_proba(car_detector.X_test)
    score = car_detector.svc.score(car_detector.X_test, car_detector.y_test)

    print(y_pred)
    print(y_prob)
    print(score)
    return


def demo2():
    detect = cv.xfeatures2d.SIFT_create(nfeatures=-1)
    bow_kmeans_trainer = cv.BOWKMeansTrainer(60)
    flann_params = dict(algorithm=1, trees=5)
    flann = cv.FlannBasedMatcher(flann_params, {})
    extract_bow = cv.BOWImgDescriptorExtractor(detect, flann)

    image_folder = '../dataset/data/car_data/TrainImages'

    filenames = os.listdir(image_folder)
    for filename in filenames:
        image_path = os.path.join(image_folder, filename)
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if image is None:
            continue
        _, descriptors = detect.detectAndCompute(image, mask=None)
        if descriptors is None:
            continue
        bow_kmeans_trainer.add(descriptors)

    vocabulary = bow_kmeans_trainer.cluster()
    extract_bow.setVocabulary(vocabulary)

    data = list()
    target = list()
    filenames = os.listdir(image_folder)
    for filename in filenames:
        image_path = os.path.join(image_folder, filename)
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        keypoints = detect.detect(image, None)
        sample_feature = extract_bow.compute(image, keypoints)
        if sample_feature is None:
            continue
        data.extend(sample_feature)
        label = 1 if filename.split('-')[0] == 'pos' else 0
        target.append(label)
    data = np.array(data)
    target = np.array(target, dtype=np.int)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    svm = SVC(kernel="linear", probability=True)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)
    score = svm.score(X_test, y_test)

    print(y_pred)
    print(y_prob)
    print(score)
    return


if __name__ == '__main__':
    demo1()
