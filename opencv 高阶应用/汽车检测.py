"""
参考文档:
http://www.yyearth.com/index.php?aid=241
数据集的下载地址:
http://www.yyearth.com/attachment.php?id=1234

BOW模型的处理过程:
1. 采用 SIFT 特征提取. 得到每张图片的特征点 keypoints 及其描述符 descriptor.
2. 汇总所有训练样本的描述符, 并采用 KMeans 聚类算法将描述符聚类成 N 个特征(到这里, 我们就可以认为所有的图片都只有这 N 种特征).
3. 计算每一张图片的特征描述符, 并映射成 KMeans 聚类得到的 N 个特征, 统计每张图片中的各特征分别出现的次数. 作为这张图的 BOW 特征(该 BOW 特征, 具有 N 个数字, 作为 1 行).
4. 至此, 每张图片都可以得到一个具有 N 个特征的向量 (如传统机器学习所需要的数据集).
5. 有了以数字表示的 BOW 特征, 则可以对图像进行传统机器学习的分类, 聚类等操作.
"""
import cv2
import numpy as np
from os.path import join

datapath = 'C:/Users/Administrator/PycharmProjects/openCV/dataset2/carData/TrainImages'

def path(cls, i):
    return "%s/%s%d.pgm" % (datapath, cls, i+1)

pos, neg = "pos-", "neg-"

detect = cv2.xfeatures2d.SIFT_create()
extract = cv2.xfeatures2d.SIFT_create()

flann_params = dict(algorithm=1, trees=5)

flann = cv2.FlannBasedMatcher(flann_params, {})

bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)

extract_bow = cv2.BOWImgDescriptorExtractor(extract, flann)

def extract_sift(fn):
    im = cv2.imread(fn, 0)
    return extract.compute(im, detect.detect(im))[1]

for i in range(8):
    bow_kmeans_trainer.add(extract_sift(path(pos, i)))
    bow_kmeans_trainer.add(extract_sift(path(neg, i)))

voc = bow_kmeans_trainer.cluster()
extract_bow.setVocabulary(voc)

def bow_features(fn):
    im = cv2.imread(fn, 0)
    return extract_bow.compute(im, detect.detect(im))

traindata, trainlabels = [], []

for i in range(20):
    traindata.extend(bow_features(path(pos, i)))
    trainlabels.append(1)
    traindata.extend(bow_features(path(neg, i)))
    trainlabels.append(-1)

svm = cv2.ml.SVM_create()

svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

def predict(fn):
    f = bow_features(fn)
    p = svm.predict(f)
    print(fn, '\t', p[1][0][0])
    return p

car = r"C:\Users\Administrator\PycharmProjects\openCV\dataset2\carData\car.jpg"

car_img = cv2.imread(car)

car_predict = predict(car)

font = cv2.FONT_HERSHEY_SIMPLEX

if car_predict[1][0][0] == 1.0:
    cv2.putText(car_img, 'Car Detected', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow('BOW + SVM Success', car_img)
cv2.waitKey(0)
cv2.destroyAllWindows()