"""
在图像中检测出关闭按钮.
主要的思路是:
通过局部图像二值化图像. 这样关闭按钮的地方就一定会显现出 X 叉的连通域.
现通过轮廓筛选这些连通域, 得到关闭按钮的位置.

轮廓筛选得方法还不够准确.
"""
import cv2 as cv
import numpy as np
import glob


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def find_contours(gray):
    """由于有的叉的线比较细, 导致 8 邻域相连的点在 4 邻域上被认为不相连, 所以出两张相反的二值图像."""
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 0)
    binary_inserse = np.array(np.where(binary == 0, 255, 0), dtype=np.uint8)

    _, contours1, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    _, contours2, _ = cv.findContours(binary_inserse, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = np.concatenate([contours1, contours2])
    if len(contours) == 0:
        return None, False
    return contours, True


def filter_contours_a(image, contours):
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    result = list()
    for contour in contours:
        contour_area = cv.contourArea(contour)
        if contour_area < 100 or contour_area > 3000:
            continue

        x, y, w, h = cv.boundingRect(contour)
        bounding_box_area = w * h
        if bounding_box_area < 300 or bounding_box_area > 10000:
            continue

        aspect_ratio = w / h
        if aspect_ratio < 0.9 or aspect_ratio > 1.1:
            continue

        area2box_rate = contour_area / bounding_box_area
        if area2box_rate > 0.5:
            continue

        # 图像在该区域的标准差不应过小.
        roi = gray[y:y+h, x:x+w]
        roi_var = np.var(roi)
        if roi_var < 300:
            continue

        # 图像像素加权的中心应接近 bounding box 的中心.
        xc = w / 2
        yc = h / 2
        moment_map = cv.moments(roi)
        m00 = moment_map["m00"]
        m01 = moment_map["m01"]
        m10 = moment_map["m10"]
        xc_ = m10 / m00
        yc_ = m01 / m00
        distance = np.sqrt((xc_ - xc)**2 + (yc_ - yc)**2)
        max_distance = np.sqrt(w ** 2 + h ** 2)
        offset_rate = distance / max_distance
        if offset_rate > 0.5:
            continue

        template1 = np.zeros(shape=(h, w))
        # 将 bounding box 看作正方形计算出的宽度.
        thickness = int((np.sqrt(w**2 + h**2) - np.sqrt(w**2 + h**2 - 3*w*h*area2box_rate)) / 3)
        cv.line(template1, pt1=(0, 0), pt2=(w, h), color=255, thickness=thickness)
        cv.line(template1, pt1=(0, h), pt2=(w, 0), color=255, thickness=thickness)
        template2 = np.zeros(shape=(h, w))
        new_contour = contour - np.array([[[x, y]]])
        cv.drawContours(template2, contours=[new_contour], contourIdx=0, color=255, thickness=-1)
        ret = template1 != template2
        error_rate = np.sum(ret) / bounding_box_area            # 建议 0.2
        # 考虑到叉的线比较细时, 在像素上的偏离可能会比较多, 按 bounding_box_area 计算动态阈值. 但是好像不太合理.
        error_rate_threshold = -0.1 * area2box_rate + 0.2
        if error_rate < error_rate_threshold:
            result.append(contour)
            print("area2box_rate", area2box_rate)
            print("error_rate", error_rate)
            # show_image(template1)
            # show_image(template2)
    if len(result) == 0:
        return None, False
    result = np.array(result)
    return result, True


def draw_rectangle_by_contours(image, contours):
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
    show_image(image)
    return


def demo1():
    image_path_list = glob.glob("dataset/*.jpg")

    for image_path in image_path_list:
        image = cv.imread(image_path)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        show_image(image)
        print(image.shape)
        contours, flag = find_contours(gray)
        if not flag:
            continue
        contours, flag = filter_contours_a(image=gray, contours=contours)
        if not flag:
            continue

        draw_rectangle_by_contours(image, contours)
    return


if __name__ == '__main__':
    demo1()
