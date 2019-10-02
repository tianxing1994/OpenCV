"""
这个东西, 效果真的很差.
不过代码的思路是理清楚了.

参考文档:
http://www.yyearth.com/index.php?aid=241
数据集的下载地址:
http://www.yyearth.com/attachment.php?id=1234
"""
import os
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class CarDetector(object):
    def __init__(self, train_image_folder):
        self._detect = cv.xfeatures2d.SIFT_create(nfeatures=-1)
        self._bow_kmeans_trainer = cv.BOWKMeansTrainer(60)
        self._extract_bow = None
        self._feature_extractor = None

        self._svc = None

        self._train_image_folder = train_image_folder
        self._vocabulary = None
        self._data = None
        self._target = None

        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None

    def _init_extract_bow(self):
        flann_params = dict(algorithm=1, trees=5)
        flann = cv.FlannBasedMatcher(flann_params, {})
        extract_bow = cv.BOWImgDescriptorExtractor(self._detect, flann)
        extract_bow.setVocabulary(self.vocabulary)
        self._extract_bow = extract_bow
        return self._extract_bow

    def _calc_vocabulary(self):
        filenames = os.listdir(self._train_image_folder)
        for filename in filenames:
            image_path = os.path.join(self._train_image_folder, filename)
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            if image is None:
                continue
            _, descriptors = self._detect.detectAndCompute(image, mask=None)
            if descriptors is None:
                continue
            self._bow_kmeans_trainer.add(descriptors)

        vocabulary = self._bow_kmeans_trainer.cluster()
        return vocabulary

    @property
    def vocabulary(self):
        if self._vocabulary is not None:
            return self._vocabulary
        self._vocabulary = self._calc_vocabulary()
        return self._vocabulary

    @property
    def extract_bow(self):
        if self._extract_bow is not None:
            return self._extract_bow
        self._extract_bow = self._init_extract_bow()
        return self._extract_bow

    def feature_extractor(self, image, mask=None):
        keypoints = self._detect.detect(image, mask)
        sample_feature = self.extract_bow.compute(image, keypoints)
        return sample_feature

    def _extract_features(self):
        data = list()
        target = list()
        filenames = os.listdir(self._train_image_folder)
        for filename in filenames:
            image_path = os.path.join(self._train_image_folder, filename)
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            sample_feature = self.feature_extractor(image, mask=None)
            if sample_feature is None:
                continue
            data.extend(sample_feature)
            label = 1 if filename.split('-')[0] == 'pos' else 0
            target.append(label)
        data = np.array(data)
        target = np.array(target, dtype=np.int)
        return data, target

    @property
    def data(self):
        if self._data is not None:
            return self._data
        self._data, self._target = self._extract_features()
        return self._data

    @property
    def target(self):
        if self._target is not None:
            return self._target
        self._data, self._target = self._extract_features()
        return self._target

    def fit_svc(self, test_size=0.0):
        if test_size == 0.0:
            self._X_train, self._y_train = self.data, self.target
        else:
            self._X_train, self._X_test, self._y_train, self._y_test = \
                train_test_split(self.data, self.target, test_size=test_size)
        svc = SVC(kernel="linear", probability=True)
        svc.fit(self._X_train, self._y_train)
        self._svc = svc
        return self._svc

    @property
    def svc(self):
        if self._svc is not None:
            return self._svc
        raise AttributeError('CarDetector object has not attribute svc.')

    @property
    def X_test(self):
        if self._X_test is not None:
            return self._X_test
        raise AttributeError('CarDetector object has not attribute X_test.')

    @property
    def y_test(self):
        if self._y_test is not None:
            return self._y_test
        raise AttributeError('CarDetector object has not attribute y_test.')

    @staticmethod
    def resize(image, scale):
        return cv.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                         interpolation=cv.INTER_AREA)

    def pyramid(self, image, scale=0.8, max_size=(500, 200), min_size=(100, 40)):
        """
        建立图像金字塔. 返回被调整过大小的图像直到宽度和高度达到规定的最小约束.
        :param image: 输入图片. 灰度图.
        :param scale: 缩放因子, 图像金字塔中每一层图片的缩小比例.
        :param max_size: (width, height), pyramid 图像金字塔底层的图片宽高不能大于 min_size 中的对应值.
        :param min_size: (width, height), pyramid 图像金字塔顶层的图片宽高不能小于 min_size 中的对应值.
        :return:
        """
        yield image
        while True:
            image = self.resize(image, scale)
            if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
                break
            yield image
        while True:
            image = self.resize(image, scale=1/scale)
            if image.shape[0] > max_size[1] and image.shape[1] > max_size[0]:
                raise StopIteration
            yield image

    @staticmethod
    def sliding_window(image, step, window_size):
        """
        :param image: 图片
        :param step: 步长
        :param window_size: (width, height). 滑动检测窗口的大小.
        :return:
        """
        h, w = image.shape[:2]
        for y in range(0, h, step):
            for x in range(0, w, step):
                roi = image[y: y + window_size[1], x: x + window_size[0]]
                if roi.shape[1] != window_size[0] or roi.shape[0] != window_size[1]:
                    raise StopIteration
                yield x, y, roi

    @staticmethod
    def non_max_suppression_fast(windows, overlap_thresh=0.25):
        """
        非极大值抑制.
        按 score 得分对窗口排序. 得分最高的窗口 window 是最应被保留的.
        而与其重叠面积比率超过 overlap_thresh 的则应被排除(抑制).
        如此迭代, 直到 windows 为空. 而被集留的 window 被存储到 pick 中.
        :param windows: ndarray, 窗口集. 如: [[x1, y1, x2, y2, score], ...].
        (x1, y1), (x2, y2) 分别表示窗口左上角/右下角坐标. score 指该窗口中包含汽车的置信度.
        :param overlap_thresh: 重叠阈值.
        :return:
        """
        pick = list()

        x1 = windows[:, 0]
        y1 = windows[:, 1]
        x2 = windows[:, 2]
        y2 = windows[:, 3]
        scores = windows[:, 4]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        indexs = np.argsort(scores)

        while len(indexs) > 0:
            picked_index = indexs[-1]
            pick.append(picked_index)

            picked_x1 = x1[picked_index]
            picked_y1 = y1[picked_index]
            picked_x2 = x2[picked_index]
            picked_y2 = y2[picked_index]
            picked_area = area[picked_index]

            # 计算每个窗口与 picked_index 窗口重叠的部分的坐标.
            overlap_x1 = np.maximum(picked_x1, x1)
            overlap_y1 = np.maximum(picked_y1, y1)
            overlap_x2 = np.minimum(picked_x2, x2)
            overlap_y2 = np.minimum(picked_y2, y2)
            overlap_w = np.maximum(0, overlap_x2 - overlap_x1 + 1)
            overlap_h = np.maximum(0, overlap_y2 - overlap_y1 + 1)
            overlap_area = overlap_w * overlap_h

            overlap_rate = overlap_area / picked_area
            negative_windows_indexs = np.where(overlap_rate > overlap_thresh)[0]
            indexs = np.delete(indexs, negative_windows_indexs)
        result = windows[pick]
        return result

    @staticmethod
    def draw_rectangle(image, windows):
        for x1, y1, x2, y2, score in windows:
            cv.rectangle(image, (int(x1), int(y1)), (int(x2)-2, int(y2)-2), (0, 255, 0), 1)
            cv.putText(image, "%f" % score, (int(x1), int(y1) - 10), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        return image

    @staticmethod
    def show_image(image, win_name='input image'):
        cv.namedWindow(win_name, cv.WINDOW_NORMAL)
        cv.imshow(win_name, image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return

    def detector_car_position(self, image, scale=0.8, window_size=(150, 80), step=1, overlap_thresh=0.25):
        """
        :param image: 输入图像, 灰度图.
        :param scale: 缩放因子, 图像金字塔中每一层图片的缩小比例. 应属于 (0, 1)
        :param window_size: (width, height). 滑动检测窗口的大小.
        :param step: int, 检测窗口滑动的步长.
        :param overlap_thresh: 非最大值抑制时, 重叠阈值.
        :return:
        """
        rectangles = list()
        h, w = image.shape
        for pyramid_image in self.pyramid(image,
                                          scale=scale,
                                          max_size=(window_size[0]*3, window_size[1]*3),
                                          min_size=window_size):
            # scale_real: 图像金字塔图像 pyramid_image 与原图的真实比例.
            scale_real = w / pyramid_image.shape[1]
            for x, y, roi in self.sliding_window(pyramid_image, step=step, window_size=window_size):
                sample_feature = self.feature_extractor(roi)
                if sample_feature is None:
                    continue
                y_prob = self.svc.predict_proba(sample_feature)[0]
                classify = np.argmax(y_prob)
                score = y_prob[1]
                if classify == 1:
                    rectangle = [x*scale_real,
                                 y*scale_real,
                                 (x+window_size[0])*scale_real,
                                 (y+window_size[1])*scale_real,
                                 score]
                    rectangles.append(rectangle)

        if len(rectangles) == 0:
            return None
        windows = np.array(rectangles)
        windows = self.non_max_suppression_fast(windows, overlap_thresh=overlap_thresh)
        return windows


def demo1():
    """封装成类."""
    train_image_folder = '../dataset/data/car_data/TrainImages'
    car_detector = CarDetector(train_image_folder=train_image_folder)
    car_detector.fit_svc(test_size=0.1)

    y_pred = car_detector.svc.predict(car_detector.X_test)
    y_prob = car_detector.svc.predict_proba(car_detector.X_test)
    score = car_detector.svc.score(car_detector.X_test, car_detector.y_test)
    print(y_pred)
    print(y_prob)
    print(score)

    test_image_path = '../dataset/data/car_data/TestImages/test-41.pgm'
    image = cv.imread(test_image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    windows = car_detector.detector_car_position(gray, scale=0.8, window_size=(100, 40), step=10, overlap_thresh=0.3)
    if windows is None:
        print('没有检测到汽车.')
        return
    print(windows)
    draw_image = car_detector.draw_rectangle(image, windows)
    car_detector.show_image(draw_image)
    return


if __name__ == '__main__':
    demo1()
