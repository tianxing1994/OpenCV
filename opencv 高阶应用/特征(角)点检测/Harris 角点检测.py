"""
参考链接:
https://www.cnblogs.com/zyly/p/9508131.html
https://github.com/makelove/OpenCV-Python-Tutorial

Harris 角点检测返回的是与原图像一样大的 ndarray,
各像素点的一个可能是角点的值. 该值越大, 则越可能是角点.

通过设定阈值, 我们会得到一些连通的区域, 再通过 cv.connectedComponentsWithStats 函数
可以求得各连接区域的中心, 可将这些中心作为真正的确点.
"""
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    image_path = '../../dataset/data/image_sample/image00.jpg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # cornerHarris参数：
    # k: Harris 角点检测方程中的自由参数,取值参数为 [0.04, 0.06].
    dst = cv.cornerHarris(src=gray, blockSize=9, ksize=23, k=0.04)

    # 变量 a 的阈值为 0.01 * dst.max(), 如果 dst 的图像值大于阈值, 那么该图像的像素点设为 True, 否则为 False
    # 将图片每个像素点根据变量 a 的 True 和 False 进行赋值处理, 赋值处理是将图像角点勾画出来.
    a = dst > (0.01 * dst.max())
    image[a] = [0, 0, 255]

    show_image(image)
    return


def demo2():
    image_path = '../../dataset/data/image_sample/image00.jpg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # cornerHarris参数：
    # ksize: 不能大于 31. 必须是奇数.
    # k: Harris 角点检测方程中的自由参数,取值参数为 [0.04, 0.06].
    dst = cv.cornerHarris(src=gray, blockSize=9, ksize=31, k=0.04)

    # 变量 a 的阈值为 0.01 * dst.max(), 如果 dst 的图像值大于阈值, 那么该图像的像素点设为 True, 否则为 False
    # 将图片每个像素点根据变量 a 的 True 和 False 进行赋值处理, 赋值处理是将图像角点勾画出来.
    result, binary = cv.threshold(src=dst, thresh=0.01 * dst.max(), maxval=255, type=cv.THRESH_BINARY)
    # show_image(binary)

    a = dst > (0.01 * dst.max())
    image[a] = [255, 0, 0]

    image_ = np.uint8(binary)
    retval, labels, stats, centroids = cv.connectedComponentsWithStats(image=image_)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # np.int0 可以用来省略小数点后的数字, 非四舍五入
    centroids_ = np.array(centroids, dtype=np.int0)
    corners_ = np.array(corners, dtype=np.int0)

    image[centroids_[:, 1], centroids_[:, 0]] = [0, 0, 255]
    image[corners_[:, 1], corners_[:, 0]] = [0, 255, 0]

    show_image(image)
    return


if __name__ == '__main__':
    demo2()
