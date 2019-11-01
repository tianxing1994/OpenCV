"""
参考链接:
https://blog.csdn.net/qq_22238021/article/details/72875087
"""
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def euclid_distance(vector1, vector2):
    return np.sqrt(np.sum(np.square(vector1 - vector2)))


def salient_hc(image, bins=50):
    h, w, c = image.shape

    image_reshape = image.reshape(-1, 3)
    image_training = shuffle(image_reshape)[:3000]
    kmeans = KMeans(n_clusters=bins, random_state=0)
    kmeans.fit(image_training)
    centers = kmeans.cluster_centers_
    index = kmeans.predict(image_reshape)

    index_ = np.array(index, dtype=np.float32)
    hist = cv.calcHist(images=[index_], channels=[0], mask=None, histSize=[bins], ranges=[0, bins])

    center_salient = list()
    for i, center_i in enumerate(centers):
        salient = 0
        for j, center_j in enumerate(centers):
            if i == j:
                continue
            d = euclid_distance(center_i, center_j)
            salient += d * hist[j]
        center_salient.append(salient)
    center_salient = np.array(center_salient)
    center_salient = center_salient / np.max(center_salient)
    salient_data = center_salient[index]

    smin = np.min(salient_data)
    smax = np.max(salient_data)
    salient_data = (salient_data - smin) / (smax - smin)
    salient_image = np.array(255 * salient_data.reshape(h, w), dtype=np.uint8)

    return salient_image


def demo1():
    """
    hc 显著性分析
    """
    # image_path = '../../dataset/data/image_sample/image00.jpg'
    image_path = '../../dataset/data/image_sample/bird.jpg'
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2LAB)

    salient_image = salient_hc(image, bins=50)
    show_image(salient_image)
    return


def demo2():
    """
    KMeans 聚类图像色彩效果展示
    """
    image_path = '../../dataset/data/image_sample/bird.jpg'
    image = cv.imread(image_path)
    image_reshape = image.reshape(-1, 3)
    image_training = shuffle(image_reshape)[:3000]
    kmeans = KMeans(n_clusters=50, random_state=0)
    kmeans.fit(image_training)
    index = kmeans.predict(image_reshape)
    centers = kmeans.cluster_centers_
    centers = np.array(centers, dtype=np.uint8)
    new_image_data = centers[index]
    new_image = new_image_data.reshape(image.shape)
    show_image(new_image)
    return


if __name__ == '__main__':
    demo1()
