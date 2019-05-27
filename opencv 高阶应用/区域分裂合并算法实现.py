"""
https://github.com/raochin2dev/RegionMerging
https://github.com/manasiye/Region-Merging-Segmentation/blob/master/Q2a.py
https://github.com/pranavgadekar/recursive-region-merging
https://github.com/CQ-zhang-2016/some-algorithms-about-digital-image-process
https://github.com/dtg67/SliceAndLabel/blob/master/SliceandLabel.py
https://blog.csdn.net/qq_19531479/article/details/79649227
"""
import cv2 as cv
import numpy as np
from scipy import signal as sp
import scipy
from PIL import Image
import scipy.ndimage as ndi
import math
import matplotlib.pyplot as plt


class Object(object):
    def __init__(self):
        self.l = 0
        self.r = 0
        self.t = 0
        self.b = 0


class minMaxPx(object):
    def __init__(self, val):
        self.min = val
        self.max = val


class ReginMerge(object):
    def __init__(self, filename):
        self.filename = filename

        self.image = cv.imread(self.filename, cv.IMREAD_GRAYSCALE)
        self.w, self.h = self.image.shape
        self.object = np.array([Object() for x in range(self.h * self.w + 1)])
        self.Label = np.array([[0 for x in range(self.h)] for y in range(self.w)])

        self.defineLabels()
        self.merge()
        self.edge = self.detectEdge()
        self.copyBackEdges()
        cv.namedWindow('image', cv.WINDOW_NORMAL)
        cv.imshow('image', self.image)

    def defineLabels(self):
        label = 255
        for i in range(self.w):
            for j in range(self.h):
                if self.Label[i, j] == 0:
                    label = label - 1
                    currInt = self.image[i, j]
                    self.object[label].l = i
                    self.object[label].r = i
                    self.object[label].t = j
                    self.object[label].b = j
                    self.recurLabel(i, j, label, currInt)

    def copyBackEdges(self):
        for i in range(self.w):
            for j in range(self.h):
                if self.edge[i, j] == 255:
                    self.image[i, j] = 255

    def detectEdge(self):
        self.output = np.zeros(self.Label.shape)
        w = self.output.shape[1]
        h = self.output.shape[0]

        for y in range(1, h-1):
            for x in range(1, w-1):
                twoXtow = self.Label[y-1:y+1, x-1:x+1]
                maxPx = twoXtow.max()
                minPx = twoXtow.min()
                Cross = False
                if minPx != maxPx:
                    Cross = True
                if Cross:
                    self.output[y, x] = 255
        return self.output

    def merge(self):
        change = 1
        iter = 0
        while change == 1:
            iter = iter + 1
            change = 0
            for i in range(self.w):
                for j in range(self.h):
                    currLabel = self.Label[i, j]
                    for x in range(i, i + 2):
                        if x >= self.w:
                            break
                        for y in range(j, j + 2):
                            if y >= self.h:
                                break
                            upLevLabel = self.Label[x, y]
                            if upLevLabel == currLabel:
                                continue
                            if self.object[upLevLabel].l < self.object[currLabel].l and \
                                    self.object[upLevLabel].r > self.object[currLabel].r and \
                                    self.object[upLevLabel].t < self.object[currLabel].t and \
                                    self.object[upLevLabel].b > self.object[currLabel].b:
                                self.recurChangeLabel(i, j, upLevLabel, currLabel)
                                change = 1
            plt.imshow(self.Label)
            plt.savefig('recurImage' + str(iter) + '.png')

    def recurChangeLabel(self, i, j, newLabel, oldLabel):
        if i<0 or i>=self.w or j<0 or j>=self.h:
            return
        if self.Label[i, j] != oldLabel:
            return
        self.Label[i, j] = newLabel
        self.recurChangeLabel(i - 1, j + 1, newLabel, oldLabel)
        self.recurChangeLabel(i, j + 1, newLabel, oldLabel)
        self.recurChangeLabel(i + 1, j - 1, newLabel, oldLabel)
        self.recurChangeLabel(i + 1, j, newLabel, oldLabel)
        self.recurChangeLabel(i + 1, j + 1, newLabel, oldLabel)

    def recurLabel(self, i, j, label, currInt):
        IntenDiff = 3.8
        if (i<0 or i>= self.w or j<0 or j>=self.h) or \
            self.Label[i, j] != 0 or \
            self.image[i, j] < currInt - IntenDiff or self.image[i, j] > currInt + IntenDiff:
            return

        currInt = self.image[i, j]
        self.Label[i, j] = label
        if self.object[label].l > i:
            self.object[label].l = i
        elif self.object[label].r < i:
            self.object[label].r = i
        if self.object[label].t > j:
            self.object[label].t = j
        elif self.object[label].b < j:
            self.object[label].b = j
        self.recurLabel(i - 1, j + 1, label, currInt)
        self.recurLabel(i, j + 1, label, currInt)
        self.recurLabel(i + 1, j + 1, label, currInt)
        self.recurLabel(i + 1, j, label, currInt)
        self.recurLabel(i + 1, j - 1, label, currInt)


if __name__ == '__main__':
    hough = ReginMerge('../dataset/lena.png')














