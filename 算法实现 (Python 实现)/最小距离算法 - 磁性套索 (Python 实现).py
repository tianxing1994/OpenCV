"""
http://code.activestate.com/recipes/117228/
https://github.com/pdyban/livewire
磁性套索的基本原理应该是：
高斯滤波或中值滤波 + 边缘提取 + 最短路径.
对读入的图像, 首先进行中值滤波, 然后进行边缘提取, 输出为一个只有边缘的灰阶图像,
然后再利用图论中的最短路径算法, 去寻找两个点之间的最短路径.
那么这个最短路径就是磁性套索中检测到的两个点区域内的边缘 (最短路径).
"""
import heapq
from math import log
import cv2 as cv
import numpy as np
import math


def shortestPath(G, start, end, length_penalty=0.0):
    def flatten(L):
        while len(L) > 0:
            yield L[0]
            L = L[1]
        return

    q = [(0, start, ())]
    visited = set()
    while True:
        (cost, v1, path) = heapq.heappop(q) # 从 heap 中取出最小的 item. 即 cost 损失最小的 item.
        if v1 not in visited:
            visited.add(v1)
            if v1 == end:
                return list(flatten(path))[::-1] + [v1]
            path = (v1, path)
            for (v2, cost2) in G[v1].items():
                if v2 not in visited:
                    heapq.heappush(q, (cost + cost2 + length_penalty * log(len(visited)), v2, path))


class LiveWireSegmentation(object):
    def __init__(self, image=None, smooth_image=False, threshold_gradient_image=False):
        super(LiveWireSegmentation, self).__init__()
        self._image = None
        self.edges = None
        self.G = None
        self.smoot_image = smooth_image
        self.threshold_gradient_image = threshold_gradient_image
        self.image = image

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self._image = value

        if self._image is not None:
            if self.smoot_image:
                self._smooth_image()

            self._compute_gradient_image()

            if self.threshold_gradient_image:
                self._threshold_gradient_image()

            self._compute_graph()
        else:
            self.edges = None
            self.G = None

    def _smooth_image(self):
        from skimage import restoration
        self._image = restoration.denoise_bilateral(self.image)

    def _compute_gradient_image(self):
        from skimage import filters
        self.edges = filters.scharr(self._image)

    def _threshold_gradient_image(self):
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(self.edges)
        self.edges = self.edges > threshold
        self.edges = self.edges.astype(float)

    def _compute_graph(self, norm_function=math.fabs):
        self.G = {}
        rows, cols = self.edges.shape
        for col in range(cols):
            for row in range(rows):
                neighbors = []
                if row > 0:
                    neighbors.append((row-1, col))
                if row < rows-1:
                    neighbors.append((row+1, col))
                if col > 0:
                    neighbors.append((row, col-1))
                if col < cols-1:
                    neighbors.append((row, col+1))

                dist = {}
                for n in neighbors:
                    dist[n] = norm_function(self.edges[row, col] - self.edges[n[0], n[1]])

                self.G[(row, col)] = dist

    def compute_shortest_path(self, from_, to_, length_penalty=0.0):
        if self.image is None:
            raise AttributeError("Load an image first!")
        path = shortestPath(self.G, from_, to_, length_penalty=length_penalty)
        return path


def show_image(image):
    cv.namedWindow('input image', cv.WINDOW_NORMAL)
    cv.imshow('input image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    image_path = '../dataset/lena.png'
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    algorithm = LiveWireSegmentation(image, smooth_image=False, threshold_gradient_image=False)
    result = algorithm.compute_shortest_path((200, 380), (150, 415), length_penalty=0.0)

    contours = np.array(result)[np.newaxis, :, :]
    # cv.drawContours(image, contours, 0, 255, 2)

    algorithm.edges[contours[0, :, 0], contours[0, :, 1]] = 255
    show_image(algorithm.edges)

    # image[contours[0, :, 0], contours[0, :, 1]] = 255
    # show_image(image)
    return


def demo2():
    image_path = '../dataset/lena.png'
    image_bgr = cv.imread(image_path)
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    algorithm = LiveWireSegmentation(image, smooth_image=True, threshold_gradient_image=True)
    result = algorithm.compute_shortest_path((200, 380), (150, 415), length_penalty=0.0)

    contours = np.array(result)[np.newaxis, :, :]
    # cv.drawContours(image, contours, 0, 255, 2)

    image_bgr[contours[0, :, 0], contours[0, :, 1], :] = (0, 0, 255)
    show_image(image_bgr)

    # image[contours[0, :, 0], contours[0, :, 1]] = 255
    # show_image(image)
    return


if __name__ == '__main__':
    demo2()

