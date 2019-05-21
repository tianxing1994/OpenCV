"""
参考文档:
http://www.yyearth.com/index.php?aid=241
数据集的下载地址:
http://www.yyearth.com/attachment.php?id=1234
相关函数:
cv2.xfeatures2d.SIFT_create
cv2.FlannBasedMatcher
cv2.BOWKMeansTrainer
cv2.BOWImgDescriptorExtractor
cv2.BOWImgDescriptorExtractor
extract.compute
bow_kmeans_trainer.add
bow_kmeans_trainer.cluster
extract_bow.setVocabulary
extract_bow.compute
cv2.ml.SVM_create
cv2.putText
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