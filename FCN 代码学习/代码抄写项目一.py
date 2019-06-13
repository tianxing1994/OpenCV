"""
参考链接:
https://github.com/shekkizh/FCN.tensorflow

说明:
好像有一个数据集, 但是下载不了, 网上也不知道去哪找.
现在, 把链接里的所有代码抄写一遍, 并做笔记. 学习理解它, 先.
"""
import numpy as np
import scipy.misc as misc


class BatchDatset(object):
    files = []
    images = []
    annotations = []
    image_options = dict()
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options=None):
        self.files = records_list
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        self._channels = True
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        self._channels = False
        self.annotations = np.array([np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])
        return

    def _transform(self, filename):
        image = misc.imread(filename)
        return













