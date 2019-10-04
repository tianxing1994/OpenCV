"""
参考链接:
https://blog.csdn.net/Tuzi294/article/details/79959199
"""
import cv2 as cv
import numpy as np


class ImageSegmentationByProject(object):
    def __init__(self):
        pass

    @staticmethod
    def _sigmentation(project, projection_noise_reduction=True, filter_length=True):
        l = len(project)
        project_mean = np.mean(project[project != 0])
        project_std = np.std(project[project != 0])
        if projection_noise_reduction:
            project_ = np.array(np.where(project > (project_mean - 1.25 * project_std), 1, 0))
        else:
            project_ = np.array(np.where(project > 0, 1, 0))

        segmentation_l = list()
        last_label = 0
        start_i = -1
        end_i = -1
        length = 0
        for i in range(0, l):
            this_label = project_[i]
            if this_label != last_label and this_label > 0:
                start_i = i
            elif this_label != last_label and this_label == 0:
                end_i = i
                length = end_i - start_i
                segmentation_l.append((start_i, end_i, length))
                start_i, end_i, length = -1, -1, 0
            else:
                pass
            last_label = this_label
        segmentation_l = np.array(segmentation_l)
        # 长度太短的去掉.
        if filter_length:
            segmentation_l_mean = np.mean(segmentation_l[:, 2])
            segmentation_l_std = np.std(segmentation_l[:, 2])
            segmentation_l = segmentation_l[segmentation_l[:, 2] > (segmentation_l_mean - segmentation_l_std)]
        return segmentation_l

    @staticmethod
    def _draw_line(image, h_segments_array, w_segments_list):
        for i in range(len(h_segments_array)):
            y1, y2, _ = h_segments_array[i]
            for j in range(len(w_segments_list[i])):
                x1, x2, _ = w_segments_list[i][j]
                cv.line(image, (x1, y1), (x1, y2), (0, 0, 255), 1)
                cv.line(image, (x1, y1), (x2, y1), (0, 0, 255), 1)
                cv.line(image, (x1, y2), (x2, y2), (0, 0, 255), 1)
                cv.line(image, (x2, y1), (x2, y2), (0, 0, 255), 1)
        return image

    def _get_segment_data(self, binary):
        h_project = np.sum(binary, axis=1)
        h_segments = self._sigmentation(h_project)
        w_segments_list = list()
        for h_segment in h_segments:
            s, e, _ = h_segment
            w_project = np.sum(binary[s: e, :], axis=0)
            w_segments = self._sigmentation(w_project,
                                      projection_noise_reduction=False,
                                      filter_length=False)
            w_segments_list.append(w_segments)
        return h_segments, w_segments_list

    def segmentation(self, binary):
        h_segments, w_segments_list = self._get_segment_data(binary)
        return h_segments, w_segments_list

    def segment_and_draw_image(self, binary, image=None):
        h_segments, w_segments_list = self._get_segment_data(binary)
        if image is not None:
            result = self._draw_line(image, h_segments, w_segments_list)
        else:
            result = self._draw_line(binary, h_segments, w_segments_list)
        return result


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def significant_image(image):
    """著于傅里叶变换的, 显著性检测"""
    dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
    dft = np.fft.fftshift(dft)

    # 幅度谱, 相位谱
    gray_spectrum = np.log(cv.magnitude(dft[:, :, 0], dft[:, :, 1]))
    phase_spectrum = np.arctan2(dft[:, :, 1], dft[:, :, 0])

    gray_spectrum_mean = cv.blur(gray_spectrum, ksize=(3, 3))
    spectral_residual = np.exp(gray_spectrum - gray_spectrum_mean)

    # 余弦谱, 正弦谱
    cos_spectrum = np.expand_dims(np.cos(phase_spectrum) * spectral_residual, axis=2)
    sin_spectrum = np.expand_dims(np.sin(phase_spectrum) * spectral_residual, axis=2)

    new_dft = np.concatenate([cos_spectrum, sin_spectrum], axis=2)
    idft = cv.dft(new_dft, cv.DFT_INVERSE)
    gray_spectrum_result = cv.magnitude(idft[:, :, 0], idft[:, :, 1])
    gray_spectrum_result = np.array(gray_spectrum_result / gray_spectrum_result.max() * 255, dtype=np.uint8)
    return gray_spectrum_result


def demo1():
    image_path = '../dataset/data/exercise_image/word.jpg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_ = significant_image(gray)
    show_image(gray_)
    _, binary = cv.threshold(gray_, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    isbp = ImageSegmentationByProject()
    result = isbp.segment_and_draw_image(binary, image)
    show_image(result)
    return


def demo2():
    image_path = '../dataset/data/exercise_image/word.jpg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    show_image(binary)

    isbp = ImageSegmentationByProject()
    result = isbp.segment_and_draw_image(binary, image)
    show_image(result)
    return


def demo3():
    image_path = '../dataset/data/exercise_image/word.jpg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    show_image(binary)

    isbp = ImageSegmentationByProject()
    result = isbp.segment_and_draw_image(binary, image)
    show_image(result)
    return


def test():
    image_path = '../dataset/data/bank_card/card_ 5.png'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 3))
    tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, rect_kernel)
    show_image(gray)
    show_image(tophat)

    gray_x = cv.Sobel(tophat, ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1)
    gray_x = np.absolute(gray_x)

    min_value, max_value = np.min(gray_x), np.max(gray_x)
    gray_x = 255 * (gray_x - min_value) / (max_value - min_value)
    gray_x = gray_x.astype('uint8')

    grad_x = cv.morphologyEx(gray_x, cv.MORPH_CLOSE, rect_kernel)
    _, binary = cv.threshold(gray_x, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # sq_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, sq_kernel)
    show_image(binary)

    isbp = ImageSegmentationByProject()
    result = isbp.segment_and_draw_image(binary, image)
    show_image(result)
    return


if __name__ == '__main__':
    demo2()

