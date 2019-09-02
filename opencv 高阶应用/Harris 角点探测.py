import cv2
import numpy as np


def harris_corners_detector():
    img = cv2.imread('../dataset/image0.JPG')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cornerHarris函数图像格式为 float32 ，因此需要将图像转换 float32 类型
    # gray = np.float32(gray)

    # cornerHarris参数：
    # src - 数据类型为 float32 的输入图像
    # blockSize - 角点检测中要考虑的邻域大小
    # ksize - Sobel 求导中使用的窗口大小
    # k - Harris 角点检测方程中的自由参数,取值参数为 [0,04,0.06].

    dst = cv2.cornerHarris(src=gray, blockSize=9, ksize=23, k=0.04)

    # 变量 a 的阈值为 0.01 * dst.max()，如果 dst 的图像值大于阈值，那么该图像的像素点设为 True，否则为 False
    # 将图片每个像素点根据变量 a 的 True 和 False 进行赋值处理，赋值处理是将图像角点勾画出来
    a = dst > (0.01 * dst.max())
    img[a] = [0, 0, 255]

    # 显示图像
    cv2.imshow('corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def sift_corners_detector():
    """
    https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
    cv2.xfeatures2d.SIFT_create() 此方法已获得专利, 因此, 此方法运行不了.
    卸载原来的 opencv, 安装:
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-contrib-python==3.4.2.16
    :return:
    """
    # 读取图片并灰度处理
    imgpath = r'../dataset/data/fruits.jpg'
    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 创建SIFT对象
    sift = cv2.xfeatures2d.SIFT_create()
    # 将图片进行SURF计算，并找出角点keypoints，keypoints是检测关键点
    # descriptor是描述符，这是图像一种表示方式，可以比较两个图像的关键点描述符，可作为特征匹配的一种方法。
    keypoints, descriptor = sift.detectAndCompute(gray, None)

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
    img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
                            color=(51, 163, 236))

    # 显示图片
    cv2.imshow('sift_keypoints', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


def surf_corners_detector():
    """
    cv2.xfeatures2d.SURF_create() 此方法已获得专利, 因此, 此方法运行不了.
    卸载原来的 opencv, 安装:
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-contrib-python==3.4.2.16
    :return:
    """
    # 读取图片并灰度处理
    imgpath = r'..\dataset\data\fruits.jpg'
    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 创建SURF对象，对象参数float(4000)为阈值，阈值越高，识别的特征越小。
    surf = cv2.xfeatures2d.SURF_create(float(400))
    # 将图片进行SURF计算，并找出角点keypoints，keypoints是检测关键点
    # descriptor是描述符，这是图像一种表示方式，可以比较两个图像的关键点描述符，可作为特征匹配的一种方法。
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
    img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                            color=(51, 163, 236))

    # 显示图片
    cv2.imshow('sift_keypoints', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    sift_corners_detector()
