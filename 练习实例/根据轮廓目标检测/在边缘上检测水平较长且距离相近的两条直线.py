import os
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image', flags=cv.WINDOW_NORMAL):
    cv.namedWindow(win_name, flags)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


class ProcessBarDetector(object):
    def __init__(self):
        self._image = None
        pass

    def _calc_edge_image(self, image):
        if image.ndim == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
        edge = cv.Canny(gray, 50, 150)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        edge = cv.dilate(edge, kernel)
        return edge

    def line_detect(self, image_edge):
        minLineLength = self._image.shape[1] * 1/2
        lines = cv.HoughLinesP(image_edge, 1, np.pi / 180, 100, minLineLength=minLineLength, maxLineGap=3)
        return lines

    def lines_spatial_filter(self, lines):
        """
        根据直线斜率, 高度上的位置, 长度, 是否在图像中间过滤直线.
        :param lines:
        :return:
        """
        if lines is None:
            return None
        h, w = self._image.shape[:2]
        result = list()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = float('inf') if x1==x2 else abs(y1-y2) / abs(x1-x2)
            length = abs(x1 - x2)
            left_length = min(x1, x2) + 1

            right_length = int(w - max([x1, x2])) + 1
            center_aligned = left_length / right_length
            if slope > 0.001:
                continue
            if y1 < h * 4/5:
                continue
            if length < w * 1/2:
                continue
            if center_aligned > 2 or 1/center_aligned > 2:
                continue
            result.append(line)
        result = np.array(result)
        return result

    def lines_matcher(self, lines, max_gap=10, max_misaligned=0.2):
        """
        应该有两条长度相当, 位置接近的直线出现, 这才是进度条.
        这样的两条直线应满足:
        1. 在高度方向的距离应小于 max_gap.
        2. 直线两端没有对齐部分的长度之和比上两直线合并长度的比值应小于.
        :param lines:
        :return:
        """
        if lines is None:
            return None

        # 1. 根据线之间高度上的距离将附合的直线分成一组.
        n = lines.shape[0]
        index = np.arange(n)
        keep1 = list()
        while index.size >= 2:
            group = list()
            i = index[-1]
            this_line = lines[i]
            other_line = lines[index[:-1]]
            gap = np.abs(other_line[:, 0, 1] - this_line[0, 1])
            matchs = np.where(gap < max_gap)
            if len(matchs) != 0:
                group.append(i)
                group.extend(index[matchs])
                keep1.append(group)
            index = index[:-1]

        # 2. 每一组线之间的最大距离不能大于 max_gap.
        keep2 = list()
        for group in keep1:
            group_lines = lines[group]
            max_y = np.max(group_lines[:, :, 1])
            min_y = np.min(group_lines[:, :, 1])
            if max_y - min_y > max_gap:
                continue
            keep2.append(group)

        # 3. 计算每组内直线对齐的长度. 取最长的保留.
        best_group = list()
        max_length = 0
        for group in keep2:
            group_lines = lines[group]
            max_x1 = np.max(group_lines[:, :, 0])
            min_x2 = np.min(group_lines[:, :, 2])
            aligned_length = min_x2 - max_x1
            if aligned_length > max_length:
                best_group = group
        if len(best_group) == 0:
            return None
        result = lines[best_group]
        return result

    def lines_filter(self, lines):
        lines = self.lines_spatial_filter(lines)
        lines = self.lines_matcher(lines)
        return lines

    @staticmethod
    def show_image(image, win_name='input image', flags=cv.WINDOW_NORMAL):
        cv.namedWindow(win_name, flags)
        cv.imshow(win_name, image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return

    def draw_lines(self, image, lines):
        if lines is None:
            return image
        for line in lines:
            x1, y1, x2, y2 = line[0]

            cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        return image

    def fit_image(self, image):
        self._image = image
        h, w = self._image.shape[:2]
        image_edge = self._calc_edge_image(image)
        show_image(image_edge)
        lines = self.line_detect(image_edge)
        if lines is None:
            return None
        lines_ = self.lines_filter(lines)
        image_with_line = self.draw_lines(image, lines_)
        show_image(image_with_line)
        return


if __name__ == '__main__':
    pbd = ProcessBarDetector()

    # 加载加, 进度条, 检测出直线, 必须要有平行的两条, 且, 两条需长度相当, 距离相近.
    # image_dir = '../../dataset/local_dataset/hepingjingying'
    image_dir = '../../dataset/local_dataset/chuanyuehuoxian'
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            image_path = os.path.join(root, file)

            image = cv.imread(image_path)
            pbd.fit_image(image)



