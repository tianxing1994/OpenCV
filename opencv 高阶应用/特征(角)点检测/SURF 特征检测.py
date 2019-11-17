import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    """
    cv2.xfeatures2d.SURF_create() 此方法已获得专利, 因此, 此方法运行不了.
    卸载原来的 opencv, 安装:
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-contrib-python==3.4.2.16
    :return:
    """
    image_path = '../../dataset/data/image_sample/lena.png'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 创建SURF对象, 对象参数 float(4000) 为阈值, 阈值越高, 识别的特征越小.
    surf = cv.xfeatures2d.SURF_create(float(4000))
    # 将图片进行 SURF 计算, 并找出角点 keypoints, keypoints 是检测到的关键点.
    # descriptor 是描述符, 这是图像一种表示方式, 可以比较两个图像的关键点描述符, 可作为特征匹配的一种方法.
    keypoints, descriptor = surf.detectAndCompute(gray, None)

    # cv2.drawKeypoints() 函数主要包含五个参数：
    # image: 原始图片
    # keypoints：从原图中获得的关键点，这也是画图时所用到的数据
    # outputimage：输出
    # color：颜色设置，通过修改（b,g,r）的值,更改画笔的颜色，b=蓝色，g=绿色，r=红色。
    # flags：绘图功能的标识设置，标识如下：
    # cv2.DRAW_MATCHES_FLAGS_DEFAULT  默认值
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    # cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
    # cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    image = cv.drawKeypoints(image=image,
                             outImage=image,
                             keypoints=keypoints,
                             flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                             color=(0, 0, 255))

    show_image(image)
    return


if __name__ == '__main__':
    demo1()
