"""
参考文档:
http://www.yyearth.com/index.php?aid=241
数据集的下载地址:
http://www.yyearth.com/attachment.php?id=1234
相关函数:


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

detect = cv2.HOGDescriptor()
extract = cv2.HOGDescriptor()

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