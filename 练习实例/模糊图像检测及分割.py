"""
参考链接:
https://github.com/WillBrennan/BlurDetection2
https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/

拉普拉斯变换
二维函数 f(x, y) 的拉普拉斯变换是分别求解其在 x 方向和 y 方向上的二阶导数并相加.
一阶导数反应的是因变量应自变量的变化速率, 而二阶导数是速率的速率.
其实也是曲线的曲率, 曲面的曲率.

容易理解, 同样的图像, 如将其大小 resize 得更小, 则曲率会变大, 更大, 则曲率会变小.
但是也许我们不能将其理解为将图像缩小, 其清晰度就变高了.
因为我们将图像放大缩小时, 按道理是应该同时对图像进行平滑操作的,
这样使得 resize 之后的图像更真实而没有锯齿感.

模糊图像检测
模糊的图像比原图在各像素点上的曲率更低 (棱角不鲜明).
因此, 对二维图像进行拉普拉斯变换, 可以得出图像在各个像素点上的曲率.
我们统计图像进行拉普拉斯变换后的值的标准差.
对于同一幅图像, 模糊的图像的标准差较小, 清晰的图像的标准差较大.

另外, 图像中模糊的位置则具有较小的曲率, 在清晰的位置则具有较大的曲率.
"""
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    # cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def fix_image_size(image, expected_pixels=2e6):
    ratio = float(expected_pixels) / float(image.shape[0] * image.shape[1])
    return cv.resize(image, (0, 0), fx=ratio, fy=ratio)


def estimate_blur(image, threshold=100):
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    blur_map = cv.Laplacian(image, cv.CV_64F)
    score = np.var(blur_map)
    return blur_map, score, bool(score < threshold)


def pretty_blur_map(blur_map, sigma=5):
    abs_image = np.log(np.abs(blur_map).astype(np.float32) + 1e-5)
    cv.blur(abs_image, (sigma, sigma))
    return cv.medianBlur(abs_image, sigma)


def demo1():
    """
    github 作者的实现, 这种方法基本上可以用于检测出模糊图像中清晰的部分.
    我不明白他对拉普拉斯变换后的操作是出于什么理论,
    也许只是针对性地想要使此图像中的清晰部分分割出来.
    """
    image_path = '../dataset/data/other/blurred_image.jpg'
    # image_path = '../dataset/data/other/panda.jpg'
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    print(image.shape)
    # image = fix_image_size(image)
    # print(image.shape)
    show_image(image)
    blur_map, score, flag = estimate_blur(image)
    print(score, flag)
    show_image(blur_map)
    result = pretty_blur_map(blur_map)
    show_image(result)
    return


def demo2():
    """
    图像模糊度检测.
    在大量的图像中, 通过设置阈值, 可以通过其模糊度进行图像是否模糊的分类.
    应该注意到, 图像放大后, 其模糊度 score 会变小.
    """
    image_path = '../dataset/data/other/blurred_image.jpg'
    # image_path = '../dataset/data/other/panda.jpg'

    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    print(image.shape)
    image = fix_image_size(image)
    show_image(image)
    print(image.shape)
    blur_map = cv.Laplacian(image, cv.CV_64F)
    score = np.var(blur_map)
    print(score)
    return


def demo3():
    image_path = '../dataset/data/other/blurred_image.jpg'
    # image_path = '../dataset/data/other/panda.jpg'
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur_map = cv.Laplacian(image, cv.CV_64F)
    abs_map = np.abs(blur_map)
    _, binary = cv.threshold(np.array(abs_map, dtype=np.uint8), 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    show_image(binary)
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    demo3()
