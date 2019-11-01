"""
照着写出来了, 跑不通.
高斯差分后的值太小, 1e-6 的数量级了差不多. 一个符合条件的极值点都检测不到.
Hissian 矩阵, 边缘响应抑制那块, 一个 True 都不返回.

参考链接:
https://github.com/rmislam/PythonSIFT/blob/master/siftdetector.py

SIFT 算法介绍
https://blog.csdn.net/u010440456/article/details/81483145
https://blog.csdn.net/lyl771857509/article/details/79675137
https://segmentfault.com/a/1190000004149225
https://zh.wikipedia.org/zh-cn/%E5%B0%BA%E5%BA%A6%E4%B8%8D%E8%AE%8A%E7%89%B9%E5%BE%B5%E8%BD%89%E6%8F%9B

SIFT 算法:


"""
import numpy as np
import cv2 as cv
from scipy.stats import multivariate_normal


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


class SiftDetector(object):
    def __init__(self, threshold=0.0, sigma=1.6, s=3, n=4):
        """
        :param threshold: 极值消除, 取值应大于 1. 原理还没明白.
        :param sigma: 在百度上说当 sigma=1.6 时, DoG 与拉普拉斯最为相似. 在参考链接中, 说 lowe 使用的是 sigma=1.6
        :param s: 用于在不同尺度下检测极值点的层数.
        由于先要高斯平滑, 再求 DoG, 且检测极值时需要在相邻两层对比, 所以实际的图像层数为 s+3.
        :param n: 指定金字塔的层数.
        """
        self._sigma = sigma
        self._s = s
        self._n = n
        self._k = 2**(1.0/s)
        self._threshold = threshold

        self._image = None
        # 高斯模糊图像金字塔
        self._pyramid_list = None
        # 高斯差分金字塔
        self._octave_list = None
        # 极值点检测标记
        self._extrema_list = None
        # 梯度幅值与方向
        self._magnitude_list = None
        self._orientation_list = None
        # 关键点
        self._keypoints = None
        self._descriptors = None

    def detect(self, image):
        self._image = image
        return self.keypoints

    def detect_and_compute(self, image):
        self._image = image
        return self.keypoints, self.descriptors

    def compute(self, keypoints):
        return

    @property
    def image(self):
        if self._image is None:
            raise AttributeError("image attribute do not exist.")
        else:
            return self._image

    def _get_scale_by_sigma(self, sigma):
        """
        在计算描述符时, 需要根据关键点中已标记的 sigma 来获取缩放尺度.
        :param sigma:
        :return:
        """
        # 对数换底公式: log_{a}(b) = ln(b) / ln(a)
        return np.log(sigma / self._sigma) / np.log(self._s)

    def _get_sigma_by_index(self, index):
        """
        根据图像在组内的索引来计算应该应用于该图像的高斯平滑 sigma.
        每组图像中同一层的图像具有相同的 sigma 值.
        :param index: 图像层在组内的索引.
        :return:
        """
        return self._sigma * self._k ** index

    def _gaussian_blur(self, image, index):
        """
        高斯平滑, 3σ 距离之外的像素都可以看作不起作用. 所以平滑窗口取 (6σ+1, 6σ+1).
        在计算 6σ+1 后, 我们需要取距离其最近的奇数作为卷积核的尺寸.
        :param image: 需要被高斯平滑的图像.
        :param index: 图像层在组内的索引
        :return:
        """
        sigma = self._get_sigma_by_index(index)
        size_ = int(6 * sigma + 1)
        size = size_ if size_ % 2 == 1 else size_ + 1
        result = cv.GaussianBlur(image, (size, size), 1.6)
        return result

    def _build_pyramid(self, image):
        """
        这里我们固定建立总共 4 层的图像金字塔, 每层再根据 s 的值得到 s+3 张, 通过不同 sigma 高斯平滑的图像.
        金字塔最底层的图像根据输入图像放大一倍得到.
        金字塔每上一层的层内第 0 张图像由前一层的索引为 self._s 的图像降采样得到.
        :param image: 应为灰度图像.
        :return:
        """
        s_ = self._s + 3
        pyramid_list = list()
        for i in range(self._n):
            if i == 0:
                image_ = cv.resize(image, dsize=(0, 0), fx=2, fy=2, interpolation=cv.INTER_LINEAR)
            else:
                image_ = cv.resize(pyramid_list[-1][:, :, self._s],
                                   dsize=(0, 0),
                                   fx=1 / 2,
                                   fy=1 / 2,
                                   interpolation=cv.INTER_LINEAR)
            h, w = image_.shape
            pyramid_ = np.zeros(shape=(h, w, s_), dtype=np.float32)
            for i in range(s_):
                pyramid_[:, :, i] = self._gaussian_blur(image_, index=i)
            pyramid_list.append(pyramid_)
        self._pyramid_list = pyramid_list
        return self._pyramid_list

    @property
    def pyramid_list(self):
        if self._pyramid_list is not None:
            return self._pyramid_list
        else:
            self._pyramid_list = self._build_pyramid(self.image)
        return self._pyramid_list

    def _build_dog(self, pyramid_list):
        """
        根据图像金字塔构建 DoG 高斯差分金字塔.
        :param pyramid_list:
        :return:
        """
        s_ = self._s + 2
        octave_list = list()
        for pyramid_ in pyramid_list:
            h, w, c = pyramid_.shape
            octave_ = np.zeros(shape=(h, w, s_), dtype=np.float32)
            for i in range(s_):
                octave_[:, :, i] = pyramid_[:, :, i + 1] - pyramid_[:, :, i]
            octave_list.append(octave_)
        self._octave_list = octave_list
        return self._octave_list

    @property
    def octave_list(self):
        if self._octave_list is not None:
            return self._octave_list
        else:
            self._octave_list = self._build_dog(self.pyramid_list)
        return self._octave_list

    @staticmethod
    def _edge_suppression(location, octave_):
        i, j, k = location
        # 求 x 方向上的导数, 即 w 宽度方向.
        dx = (octave_[i, j+1, k] - octave_[i, j-1, k]) / 2 / 255
        dy = (octave_[i+1, j, k] - octave_[i-1, j, k]) / 2 / 255
        dz = (octave_[i, j, k+1] - octave_[i, j, k-1]) / 2 / 255

        # 求二阶导数.
        dxx = (octave_[i, j+1, k] + octave_[i, j-1, k] - 2 * octave_[i, j, k]) / 255
        dyy = (octave_[i+1, j, k] + octave_[i-1, j, k] - 2 * octave_[i, j, k]) / 255
        dzz = (octave_[i, j, k+1] + octave_[i, j, k-1] - 2 * octave_[i, j, k]) / 255

        dxy = (octave_[i+1, j+1, k] + octave_[i-1, j-1, k] - octave_[i+1, j-1, k] - octave_[i-1, j+1, k]) / 4 / 255
        dxz = (octave_[i, j+1, k+1] + octave_[i, j-1, k-1] - octave_[i, j+1, k-1] - octave_[i, j-1, k+1]) / 4 / 255
        dyz = (octave_[i+1, j, k+1] + octave_[i-1, j, k-1] - octave_[i+1, j, k-1] - octave_[i-1, j, k+1]) / 4 / 255

        dd = np.array([[dx],
                       [dy],
                       [dz]])

        h = np.matrix([[dxx, dxy, dxz],
                       [dxy, dyy, dyz],
                       [dxz, dyz, dzz]])

        x_hat = np.linalg.lstsq(h, dd, rcond=-1)[0]
        d_x_hat = octave_[i, j, k] + 0.5 * np.dot(dd.T, x_hat)

        r = 10.0

        if (((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * ((r + 1) ** 2) and \
                (np.absolute(x_hat[0]) < 0.5) and \
                (np.absolute(x_hat[1]) < 0.5) and \
                (np.absolute(x_hat[2]) < 0.5) and \
                (np.absolute(d_x_hat) > 0.03):
            return True
        else:
            return False

    def _extrema_detect(self, octave_list):
        s_ = self._s + 2
        extrema_list = list()
        for octave_ in octave_list:
            h, w, c = octave_.shape
            extrema_ = np.zeros(shape=(h, w, self._s), dtype=np.float32)
            for i in range(1, h-1):
                for j in range(1, w-1):
                    for k in range(2, c-2):
                        value = octave_[i, j, k]
                        # DoG 高斯差分值, 小于一定阈值的点不考虑作为极值点响应.
                        if np.absolute(value) <= self._threshold:
                            continue
                        box = octave_[i-1:i+2, j-1:j+2, k-1:k+2]

                        # 每个金字塔的元素在空间和范围上大于或小于其26个直接邻居的元素都标记为极值.
                        # 如 Lowe 论文的第 4 节中所述, 通过检查它们的对比度和曲率是否超过特定阈值来修剪这些初始极值.
                        if value == box.max() or value == box.min():
                            # 因为 extrema_ 的第三个维度比 octave_ 在两端分别少一个, 所以 octave_ 中的层索引应减 1.
                            if self._edge_suppression(location=(i, j, k), octave_=octave_):
                                extrema_[i, j, k-1] = 1
            extrema_list.append(extrema_)

        self._extrema_list = extrema_list
        print(f"Number of extrema in first octave: {np.sum(extrema_list[0])}")
        print(f"Number of extrema in second octave: {np.sum(extrema_list[1])}")
        print(f"Number of extrema in third octave: {np.sum(extrema_list[2])}")
        print(f"Number of extrema in fourth octave: {np.sum(extrema_list[3])}")
        return self._extrema_list

    @property
    def extrema_list(self):
        if self._extrema_list is not None:
            return self._extrema_list
        else:
            self._extrema_list = self._extrema_detect(self.octave_list)
        return self._extrema_list

    def _gradient_magnitude_orientation(self, pyramid_list):
        """
        计算每个图像中每个像素点的梯度大小, 梯度方向.
        图像金字塔的每层中有 s+3 张图像. 此处只计算索引为 0, 1, 2 的三层图像.
        :param pyramid_list:
        :return:
        """
        magnitude_list = list()
        orientation_list = list()
        for pyramid in pyramid_list:
            h, w, c = pyramid.shape
            magnitude = np.zeros(shape=(h, w, self._s))
            orientation = np.zeros(shape=(h, w, self._s))
            for i in range(1, h-1):
                for j in range(1, w-1):
                    for k in range(self._s):
                        magnitude[i, j, k] = np.sqrt((pyramid[i+1, j, k] - pyramid[i-1, j, k]) ** 2 +
                                                     (pyramid[i, j+1, k] - pyramid[i, j-1, k]) ** 2)
                        # 计算梯度方向, 以 x 轴, 宽度方向的梯度值 dx 与 y 轴, 高度方向的梯度值 dy,
                        # 计算梯度方向, 表示为弧度. 弧度乘以 180/pi 可以转换为角度.
                        orientation[i, j, k] = (18 / np.pi) * np.arctan2((pyramid[i, j+1, k] - pyramid[i, j-1, k]),
                                                                         (pyramid[i+1, j, k] - pyramid[i-1, j, k]))
            magnitude_list.append(magnitude)
            orientation_list.append(orientation)
        self._magnitude_list = magnitude_list
        self._orientation_list = orientation_list
        return self._magnitude_list, self._orientation_list

    @property
    def magnitude_list(self):
        if self._magnitude_list is not None:
            return self._magnitude_list
        else:
            self._magnitude_list, self._orientation_list = self._gradient_magnitude_orientation(self.pyramid_list)
        return self._magnitude_list

    @property
    def orientation_list(self):
        if self._orientation_list is not None:
            return self._orientation_list
        else:
            self._magnitude_list, self._orientation_list = self._gradient_magnitude_orientation(self.pyramid_list)
        return self._orientation_list

    def _calc_keypoint(self):
        keypoints_total = 0
        for extrema in self.extrema_list:
            keypoints_total += np.sum(extrema)
        if keypoints_total == 0:
            raise ValueError(f"the detected keypoints total could not be: {keypoints_total}")
        keypoints = np.zeros(shape=(keypoints_total, 4))
        keypoints_count = 0
        for index, (magnitude, orientation, extrema) in enumerate(zip(self.magnitude_list,
                                                                      self.orientation_list,
                                                                      self.extrema_list)):

            h, w, c = magnitude.shape
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        if extrema[i, j, k] != 1:
                            continue
                        sigma = self._get_sigma_by_index(k)
                        # multivariate_normal 用于构建一个高斯核密度函数, 以计算各点的权重.
                        gaussian_window = multivariate_normal(mean=(0, 0), cov=((1.5 * sigma) ** 2))
                        # 在关键点半径为 window_radius 范围内的像素的梯度方向被考虑.
                        radius = int(np.floor(1.5 * sigma * 3))
                        # 将梯度方向分成 36 柱, 每柱 10 度.
                        orientation_hist = np.zeros(shape=(36, 1))
                        for x in range(-radius, radius+1):
                            ylim = np.sqrt(radius ** 2 - x ** 2)
                            for y in range(-ylim, ylim+1):
                                # 不在 h, w 范围内的不考虑.
                                if j+x < 0 or j+x > w-1 or i+y < 0 or i+y > h-1:
                                    continue
                                # 根据梯度幅值与概率密度值作为权重为每个梯度方向加权.
                                weight = magnitude[i+y, j+x, k] * gaussian_window.pdf(x=(x, y))
                                bin_index = int(np.clip(np.floor(orientation[i+y, j+x, k]), 0, 35))
                                orientation_hist[bin_index] += weight
                        # 取该关键点的主方向.
                        max_value = np.max(orientation_hist)
                        max_index = np.argmax(orientation_hist)
                        # 四个值, 分别存关键点的坐标索引, 所对应的标准差, 对应的主方向.
                        # 其需要将坐标还原到在原图像中的像素级坐标, 以及相对于原图像的标准差.
                        # 作为该关键字的位置和尺度的明确标记.
                        # 虽然计算关键点方向时使用的仍是根据层内索引得到的标准差, 那是因为其对应的尺度是已经缩小后的.
                        scale = 0.5 * 2 ** index
                        scale_sigma = self._get_sigma_by_index(index * self._s + k)
                        keypoints[keypoints_count, :] = np.array([int(i * scale), int(j * scale), scale_sigma, max_index])
                        keypoints_count += 1

                        orientation_hist[max_index] = 0
                        new_max_value = np.max(orientation_hist)
                        # 大于主方向 80% 的方向作为一个辅方向, 定义成一个新的关键点存在.
                        while new_max_value > 0.8 * max_value:
                            new_max_index = np.argmax(orientation_hist)
                            np.append(keypoints, np.array([[int(i * scale), int(j * scale), scale_sigma, max_index]]), axis=0)
                            orientation_hist[new_max_index] = 0
                            new_max_value = np.max(orientation_hist)
        self._keypoints = keypoints
        return self._keypoints

    @property
    def keypoints(self):
        if self._keypoints is not None:
            return self._keypoints
        else:
            self._keypoints = self._calc_keypoint()
        return self._keypoints

    def _calc_descriptor(self, keypoints):
        h, w, _ = self.image.shape
        magnitude_ = np.zeros(shape=(h, w, self._s * self._n))
        orientation_ = np.zeros(shape=(h, w, self._s * self._n))
        # 将 self._magnitude_list, self._orientation_list 的各层大小缩放到原图像的大小 (h, w).
        for index, (magnitude, orientation) in enumerate(zip(self.magnitude_list, self.orientation_list)):
            for k in range(self._s):
                k_ = self._s * index + k
                mag_max = np.max(magnitude[:, :, k])
                magnitude_[:, :, k_] = cv.resize(magnitude[:, :, k], dsize=(w, h), interpolation=cv.INTER_LINEAR)
                magnitude_[:, :, k_] = magnitude_[:, :, k_] * mag_max / np.max(magnitude_[:, :, k_])
                orientation_[:, :, k_] = cv.resize(orientation_[:, :, k], dsize=(w, h), interpolation=cv.INTER_LINEAR)
                orientation_[:, :, k_] = np.array(orientation_[:, :, k_] * 35 / np.max(orientation_[:, :, k_]),
                                                  dtype=np.int)
        descriptors = np.zeros([keypoints.shape[0], 128])
        for i in range(keypoints.shape[0]):
            for x in range(-8, 8):
                for y in range(-8, 8):
                    theta = 10 * keypoints[i, 3] * np.pi / 180.0
                    m = np.array([[np.cos(theta), - np.sin(theta)],
                                  [np.sin(theta), np.cos(theta)]])
                    xrot, yrot = np.dot(m, np.array([x, y]).T)
                    scale_index = self._get_scale_by_sigma(keypoints[i, 2])
                    x0 = keypoints[i, 0]
                    y0 = keypoints[i, 1]
                    gaussian_window = multivariate_normal(mean=(0, 0), cov=8)
                    weight = magnitude_[int(x0 + xrot), int(y0 + yrot), scale_index] * \
                             gaussian_window.pdf([xrot, yrot])
                    # 计算该点与主方向的夹角
                    angle = orientation_[int(x0 + xrot), int(y0 + yrot), scale_index] - keypoints[i, 3]
                    if angle < 0:
                        angle = 35 + angle
                    # 将角度分成 8 个方向.
                    bin_idx = np.clip(np.floor((8.0 / 35) * angle), 0, 7).astype(int)
                    # -8 到 8. 宽高 16 的范围内每距离 4 分成一个方块, 一共分成 16 个子块. 每个子块有 8 个方向.
                    block_x = int((x + 8) / 4)
                    block_y = int((y + 8) / 4)
                    descriptors[i, (4 * block_x + block_y) * 8 + bin_idx] += weight
            # 描述子的每个值, 归一化到 0-1 之间
            descriptors[i, :] = descriptors[i, :] / np.linalg.norm(descriptors[i, :])
            # 描述子的每个值, 剪切到 0 - 0.2 之间
            descriptors[i, :] = np.clip(descriptors[i, :], 0, 0.2)
            # 描述子的每个值, 重新归一化到 0-1 之间
            descriptors[i, :] = descriptors[i, :] / np.linalg.norm(descriptors[i, :])
        self._descriptors = descriptors
        return self._descriptors

    @property
    def descriptors(self):
        if self._descriptors is not None:
            return self._descriptors
        else:
            self._descriptors = self._calc_descriptor(self.keypoints)
        return self._descriptors


if __name__ == '__main__':
    image_path = "../../dataset/data/other_sample/box_in_scene.png"
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    sift = SiftDetector()
    result = sift.detect(image = gray)
    print(result)
