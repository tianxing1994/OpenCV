import cv2 as cv
import numpy as np


def show_image(image, win_name='input image', flags=cv.WINDOW_NORMAL):
    cv.namedWindow(win_name, flags)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def show_contour(contour, canvas_size=(640, 360)):
    contour_area = cv.contourArea(contour)
    print(contour_area)
    x, y, w, h = cv.boundingRect(contour)
    if w * h < 500:
        return
    box = np.zeros(shape=canvas_size, dtype=np.uint8)
    cv.drawContours(box, [contour, ], 0, 255, -1)
    show_image(box, flags=cv.WINDOW_AUTOSIZE)
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


def convex_hull_rectangular_detection(contour, min_hull_area=3000, max_hull_area=10000, min_fill_rate=0.9):
    hull = cv.convexHull(contour, returnPoints=True)
    hull_area = cv.contourArea(hull)
    x, y, w, h = cv.boundingRect(hull)
    bbox_area = w * h
    fill_rate = hull_area / bbox_area
    if hull_area < min_hull_area:
        return False
    if hull_area > max_hull_area:
        return False
    if fill_rate < min_fill_rate:
        print(f"convex_hull_rectangular fill_rate not satisfied: {fill_rate}")
        return False
    return True


def rectangular_detection(contour, threshold=0.9):
    """
    计算矩形度量. 用轮廓面积/最小外接 bounding box 面积. 比率不应小于 threshold
    :param contour:
    :param threshold: 矩形度量阈值, 轮廓面积/最小外接 bounding box 面积, 不应小于该值.
    :return:
    """
    if convex_hull_rectangular_detection(contour):
        return True
    area = cv.contourArea(contour)
    x, y, w, h = cv.boundingRect(contour)
    bbox_area = w * h
    fill_rate = area / bbox_area
    if fill_rate < threshold:
        print(f"rectangular fill_rate not satisfied: {fill_rate}")
        return False
    return True


def convex_hull_roundness_detection(contour, min_hull_area=3000, max_hull_area=10000, min_fill_rate=0.9):
    hull = cv.convexHull(contour, returnPoints=True)
    center, radius = cv.minEnclosingCircle(hull)
    circle_area = np.pi * radius ** 2
    hull_area = cv.contourArea(hull)
    fill_rate = hull_area / circle_area
    if hull_area < min_hull_area:
        return False
    if hull_area > max_hull_area:
        return False
    if fill_rate < min_fill_rate:
        print(f"convex_hull_roundness fill_rate not satisfied: {fill_rate}")
        return False
    return True


def roundness_detection(contour, threshold=0.9):
    """
    计算圆度. 用轮廓面积/最小外接圆面积. 比率不应小于 threshold
    :param contour:
    :param threshold: 圆度阈值, 轮廓面积/最小外接圆面积, 不应小于该值.
    :return: True 或 False.
    """
    if convex_hull_roundness_detection(contour):
        return True
    center, radius = cv.minEnclosingCircle(contour)
    circle_area = np.pi * radius ** 2
    contour_area = cv.contourArea(contour)
    fill_rate = contour_area / circle_area
    if fill_rate < threshold:
        print(f"roundness fill_rate not satisfied: {fill_rate}")
        return False
    return True


def shape_filter(contour):
    """
    形状过滤器, 验证轮廓是否满足任意一种几何形状.
    :param contour:
    :return:
    """
    if rectangular_detection(contour):
        return True
    if roundness_detection(contour):
        return True
    if convexity_defects_filter(contour):
        return True
    return False


def convexity_defects_filter(contour, min_vertices_num=4, min_contour_area=100, min_hull_area=3000,
                             max_defect_distance=800, max_defect_area=1000, max_defect_rate=0.08):
    """
    凸包缺陷检测,
    :param contour: 输入轮廓
    :param min_vertices_num: 轮廓最小顶点数.
    :param min_contour_area: 轮廓最小面积.
    :param min_hull_area: 最小凸包面积.
    :param max_defect_distance: 单个凸包缺陷的最大距离.
    :param max_defect_area: 单个凸包缺陷的最大面积.
    :param max_defect_rate: 凸包缺陷率最大值. 凸包缺陷的总面积与轮廓面积的比值的最大值.
    :return: True 或 False.
    """
    if contour.shape[0] < min_vertices_num:
        return False
    contour_area = cv.contourArea(contour)
    if contour_area < min_contour_area:
        return False
    hull = cv.convexHull(contour, returnPoints=False)
    hull_ = cv.convexHull(contour, returnPoints=True)
    hull_area = cv.contourArea(hull_)
    if hull_area < min_hull_area:
        return False
    hull_defects = cv.convexityDefects(contour, convexhull=hull, convexityDefects=None)
    if hull_defects is None:
        print(f"hull_defects not satisfied: {hull_defects}")
        return False

    total_defect_area = 0
    length = len(hull_defects)
    for i in range(length):
        data = hull_defects[i][0]
        index1 = data[0]
        index2 = data[1]
        index3 = data[2]
        distance = data[3] / 256.0
        if distance > max_defect_distance:
            print(f"distance not satisfied: {distance}")
            return False
        point1 = contour[index1][0]
        point2 = contour[index2][0]
        d = calc_distance(point1, point2)
        defect_area = 0.5 * d * distance
        total_defect_area += defect_area
        if defect_area > max_defect_area:
            print(f"defect_area not satisfied: {defect_area}")
            return False
    defect_rate = total_defect_area / contour_area
    if defect_rate > max_defect_rate:
        print(f"defect_rate not satisfied: {defect_area}")
        return False
    return True


def vertices_number_filter(contour):
    if contour.shape[0] < 4:
        return False
    return True


def contour_area_filter(contour):
    area = cv.contourArea(contour)
    if area < 100:
        return False
    return True


def contour_filter(contours):
    """过滤掉不符合条件的轮廓"""
    result = list()
    for i, contour in enumerate(contours):
        if not vertices_number_filter(contour):
            continue
        if not contour_area_filter(contour):
            continue
        # show_contour(contour)
        # if not convexity_defects_filter(contour):
        #     continue
        if not shape_filter(contour):
            continue
        result.append(contour)
    return result


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


def demo3():
    # image_path = '../dataset/local_dataset/snapshot_1572513078.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572427571.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572428454.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572483504.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572330104.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572424919.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572426537.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572427350.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572483700.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572489017.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572492362.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572503782.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572509515.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572510101.jpg'
    image_path = '../dataset/local_dataset/snapshot_1572514220.jpg'
    image_origin = cv.imread(image_path)
    image, h_radio, w_radio = resize_image(image_origin)

    binary = calc_bright_binary(image)
    show_image(binary)
    _, contours, heriachy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = contour_filter(contours)
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), 2)

    show_image(image)
    return


if __name__ == '__main__':
    demo3()

