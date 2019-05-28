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


def show_image(image):
    cv.namedWindow('input image', cv.WINDOW_NORMAL)
    cv.imshow('input image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


class Block(object):
    """将图像分割成许多块, l, r, t, b 分别为块的最左, 右, 上, 下的界线. """
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
    def __init__(self):
        pass

    def run(self, filename):
        self.image = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        self.w, self.h = self.image.shape

        # 图像有 w * h 个像素, 此处创建 w*h 个 Block 实例(图像最多可以被分割成 w*h 个块).
        # 图像分割时, 每一个块对应同一个 block 对象, 并由该 block 对象记录该块的信息.
        self.block = np.array([Block() for i in range(self.h * self.w)])

        # 一张大小与 image 相同的标签图, 用于标注每个像素属于哪个 block 对象.
        self.label = np.zeros(shape=self.image.shape)

        # [图像分割] 遍历每个像素, 为每个像素指定一个标签. 具体方法, 见函数.
        self.define_labels()

        # [图像合并]
        self.merge()

        self.block_edge_map = self.detect_edge()

        self.copyBackEdges()

        show_image(self.image)
        return

    def define_labels(self):
        """
        从左上向右下, 遍历图像中的每一个像素, 将该像素标记为块 label,
        并根据该像素向其8邻域的方向扩散该块, 扩散的标准为邻域的像素值与当
        前像素值的差值小于 4.
        :return:
        """
        label = 0
        for i in range(self.w):
            for j in range(self.h):
                # 遍历 label 图中的每一个对应像素, 如果 label[i, j] == 0, 则说明该像不还未被分配.
                # 此处为每个对应像素分配 block 标签.
                if self.label[i, j] == 0:
                    current_pixel = self.image[i, j]

                    # label 是这个块的标记数字, 从 self.block 索引出的对象则是分配给该块的对象.
                    # l, r, t, b 分别代表该块在 x 轴的最左最右像素坐标, 在 y 轴的最上最下像素坐标.
                    # 因此每次有像素被加入到该块时, 都应检查更新 lrtb 值, 以确保它们是自己方向上的极值.
                    self.block[label].l = i
                    self.block[label].r = i
                    self.block[label].t = j
                    self.block[label].b = j
                    # 已经初始化了该块的第一个像素, 下面的函数, 递归地遍历该像素的邻域像素,
                    # 如果相邻像素的灰度差值小于 threshold,
                    # 则将邻域的像素也标记为当前块的 label. 并更新当前块的 lrtb 值.
                    self.recursively_label(i, j, label, current_pixel)

                    label += 1
        return

    def copyBackEdges(self):
        for i in range(self.w):
            for j in range(self.h):
                if self.block_edge_map[i, j] == 255:
                    self.image[i, j] = 255

    def detect_edge(self):
        """
        遍历 label 图像中的每一个像素位置, 如果以当前像素为右下角像素的相邻 2*2 四像素中存在不同标签值.
        则说明当前像素是一个块边界像素, 将其值设为 255, 最终得到所有的块边缘轮廓的图像.
        :return:
        """
        output = np.zeros(self.label.shape)

        w = output.shape[1]
        h = output.shape[0]

        for y in range(1, h-1):
            for x in range(1, w-1):
                twoXtow = self.label[y-1:y+1, x-1:x+1]
                maxPx = twoXtow.max()
                minPx = twoXtow.min()
                if minPx != maxPx:
                    output[y, x] = 255
        return output

    def merge(self):
        """
        对于图像中的每一个像素, 获取其所属的块标签, 检测以该像素为左上角像素的 2*2 四个像素.
        如果除了当前像素的另三个像素不与当前像素一样是同一个块标签.
        且, 其所属块的 lrtb 值都比当前像素更靠近中心. 则沿着相邻像素递归地搜索. 将该块标签替换当前块标签.
        :return:
        """
        flag = 1
        count = 0
        while flag == 1:
            count += 1
            flag = 0
            # 如果遍历了所有的像素后, 存在任意一个像素被重新分配了标签, 则将 flag 置为 1, 即, 继续从头遍历图像.
            # 直到遍历了所有的像素后, 没有任意一个像素被重新分配, 才停下来.
            for i in range(self.w):
                for j in range(self.h):
                    current_label = int(self.label[i, j])
                    for x in range(i, i + 2):
                        if x >= self.w:
                            break
                        for y in range(j, j + 2):
                            if y >= self.h:
                                break
                            check_label = int(self.label[x, y])
                            if check_label == current_label:
                                continue
                            if self.block[check_label].l < self.block[current_label].l and \
                                    self.block[check_label].r > self.block[current_label].r and \
                                    self.block[check_label].t < self.block[current_label].t and \
                                    self.block[check_label].b > self.block[current_label].b:
                                self.recursively_update_label(i, j, check_label, current_label)
                                flag = 1
            plt.imshow(self.label)
            # plt.savefig('recurImage' + str(count) + '.png')

    def recursively_update_label(self, i, j, new_label, old_label):
        if i<0 or i>=self.w or j<0 or j>=self.h:
            return
        if self.label[i, j] != old_label:
            return
        self.label[i, j] = new_label
        self.recursively_update_label(i - 1, j + 1, new_label, old_label)
        self.recursively_update_label(i, j + 1, new_label, old_label)
        self.recursively_update_label(i + 1, j + 1, new_label, old_label)
        self.recursively_update_label(i + 1, j - 1, new_label, old_label)
        self.recursively_update_label(i + 1, j, new_label, old_label)

    def recursively_label(self, i, j, label, current_pixel):
        """
        根据 current_pixel 的值及其所处的 i, j 坐标位置, 检查其 8 邻域内的像素灰度值.
        如果差值小于 threshold. 则将其邻域像素也标记为该块.
        由于图像分割标记的顺序是至左向右, 至上向下. 所以对于每个未知像素,
        其左和上方向的像素都必然是已经标记过的.
        所以此函数只处理 (i-1, j+1), (i, j+1), (i+1, j+1), (i+1, j), (i+1, j-1) 5 个邻域像素.
        如果其邻域像素满足了合并条件, 则会根据其, 递归地调用此函数.
        :param i:
        :param j:
        :param label:
        :param current_pixel:
        :return:
        """
        # threshold = 3.8
        # if (i<0 or i>= self.w or j<0 or j>=self.h) or \
        #     self.label[i, j] != 0 or \
        #     abs(int(self.image[i, j]) - int(current_pixel)) > threshold:
        #     # 像素合并的条件. 不满足则直接返回.
        #     return
        #
        # # 函数运行到此则说明其满足合并条件.
        # # 更新当前像素值, block 块信息, 并递归地调用自身.
        # current_pixel = self.image[i, j]
        # self.label[i, j] = label
        # if self.block[label].l > i:
        #     self.block[label].l = i
        # elif self.block[label].r < i:
        #     self.block[label].r = i
        # if self.block[label].t > j:
        #     self.block[label].t = j
        # elif self.block[label].b < j:
        #     self.block[label].b = j
        # self.recursively_label(i - 1, j + 1, label, current_pixel)
        # self.recursively_label(i, j + 1, label, current_pixel)
        # self.recursively_label(i + 1, j + 1, label, current_pixel)
        # self.recursively_label(i + 1, j, label, current_pixel)
        # self.recursively_label(i + 1, j - 1, label, current_pixel)

        threshold = 4
        queue_list = []
        index = (i, j)
        queue_list.append(index)
        while True:
            if len(queue_list) == 0:
                return

            i, j = queue_list.pop()
            if (i < 0 or i >= self.w or j < 0 or j >= self.h) or \
                    self.label[i, j] != 0 or \
                    abs(int(self.image[i, j]) - int(current_pixel)) > threshold:
                return

            current_pixel = self.image[i, j]
            self.label[i, j] = label
            if self.block[label].l > i:
                self.block[label].l = i
            elif self.block[label].r < i:
                self.block[label].r = i
            if self.block[label].t > j:
                self.block[label].t = j
            elif self.block[label].b < j:
                self.block[label].b = j
            queue_list.extend([(i-1, j+1), (i, j+1), (i+1, j+1), (i+1, j), (i+1, j-1)])



if __name__ == '__main__':
    regin_merge = ReginMerge()
    regin_merge.run(filename='../dataset/lena.png')














