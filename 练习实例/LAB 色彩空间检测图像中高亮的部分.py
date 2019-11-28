import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def resize_image(image):
    """
    将图像 resize 到固定大小, 并返回在宽度与高度上的比例
    需要适应原图的长宽比例, 原图长边 resize 后依然是长边.
    :param image:
    :return:
    """
    fixed_size = (360, 640)
    h, w = image.shape[:2]
    if h > w:
        rh, rw = fixed_size[::-1]
    else:
        rh, rw = fixed_size
    h_radio = rh / h
    w_radio = rw / w
    image_resized = cv.resize(image, dsize=(0, 0), fx=w_radio, fy=h_radio)
    return image_resized, h_radio, w_radio


def calc_distance(point1, point2):
    result = np.sqrt(np.sum(np.square(point1 - point2)))
    return result


def bounding_box_rate_filter(contour):
    area = cv.contourArea(contour)
    x, y, w, h = cv.boundingRect(contour)
    bbox_area = w * h
    fill_rate = area / bbox_area
    if fill_rate < 0.8:
        # print(f"fill_rate not satisfied: {fill_rate}")
        return False
    return True


def convexity_defects_filter(contour):
    hull = cv.convexHull(contour, returnPoints=False)
    hull_defects = cv.convexityDefects(contour, convexhull=hull, convexityDefects=None)
    if hull_defects is None:
        return False

    length = len(hull_defects)
    for i in range(length):
        data = hull_defects[i][0]
        index1 = data[0]
        index2 = data[1]
        index3 = data[2]
        distance = data[3]
        if distance > 1000:
            # print(f"distance not satisfied: {distance}")
            return False
        point1 = contour[index1][0]
        point2 = contour[index2][0]
        d = calc_distance(point1, point2)
        defect_area = 0.5 * d * distance
        if defect_area > 100000:
            return False
    return True


def contour_area_filter(contour):
    area = cv.contourArea(contour)
    if area < 1000:
        # print(f"contour area not satisfied: {area}")
        return False
    return True


def calc_bright_binary(image):
    """
    将图像转换到 LAB 空间, 取 L 亮度通道计算二值图.
    最后通过 Canny 边缘将二值图进行分割, 避免一些部分的粘连.
    :param image:
    :return:
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l_image = lab_image[:, :, 0]

    edge = cv.Canny(gray, 50, 150)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    edge_dilate = cv.dilate(edge, kernel)

    _, binary = cv.threshold(l_image, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    index = np.where(edge_dilate == 255)
    binary[index] = 0
    return binary


def contour_filter(contours):
    """过滤掉不符合条件的轮廓"""
    result = list()
    for i, contour in enumerate(contours):
        if not convexity_defects_filter(contour):
            continue
        if not bounding_box_rate_filter(contour):
            continue
        if not contour_area_filter(contour):
            continue
        result.append(contour)
    return result


def demo3():
    # image_path = '../dataset/local_dataset/snapshot_1572513078.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572427571.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572428454.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572483504.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572330104.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572424919.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572426537.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572427350.jpg'
    image_path = '../dataset/local_dataset/snapshot_1572483700.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572489017.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572492362.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572503782.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572509515.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572510101.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572514220.jpg'
    image_origin = cv.imread(image_path)
    image, h_radio, w_radio = resize_image(image_origin)

    binary = calc_bright_binary(image)
    _, contours, heriachy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = contour_filter(contours)
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), -1)

    show_image(image)
    return


if __name__ == '__main__':
    demo3()

