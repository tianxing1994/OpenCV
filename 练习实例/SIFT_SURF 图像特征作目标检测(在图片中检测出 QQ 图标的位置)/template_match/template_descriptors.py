import os
import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans

from template_match.config import PROJECT_PATH


class DescriptorsManager(object):
    def __init__(self, dataset, n_clusters=300):
        self._dataset = dataset
        self._n_clusters = n_clusters

        self._sift = cv.xfeatures2d.SIFT_create()

        self._origin_descriptors = list()
        self._kmeans_descriptors = None

    def _init_origin_descriptors(self):
        for sample in self._dataset:
            relative_path, boxes, cls, channel = sample
            image_path = os.path.join(PROJECT_PATH, 'template_match', relative_path)
            image = cv.imread(image_path)
            for box in boxes:
                x, y, w, h = box
                roi = image[y:y+h, x:x+w]
                _, descriptors = self._sift.detectAndCompute(roi, None)
                self._origin_descriptors.extend(descriptors)
        return self._origin_descriptors

    @property
    def origin_descriptors(self):
        if len(self._origin_descriptors) != 0:
            return self._origin_descriptors
        self._origin_descriptors = self._init_origin_descriptors()
        return self._origin_descriptors

    def _init_kmeans_descriptors(self):
        x = np.array(self.origin_descriptors)
        kmeans = KMeans(n_clusters=self._n_clusters, random_state=0)
        kmeans.fit(x)
        self._kmeans_descriptors = kmeans.cluster_centers_
        return self._kmeans_descriptors

    @property
    def kmeans_descriptors(self):
        if self._kmeans_descriptors is not None:
            return self._kmeans_descriptors
        self._kmeans_descriptors = self._init_kmeans_descriptors()
        return self._kmeans_descriptors
