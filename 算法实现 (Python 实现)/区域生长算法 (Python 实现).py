"""
https://blog.csdn.net/weixin_40419806/article/details/80971205

区域生长实现的步骤如下:
1. 对图像顺序扫描!找到第1个还没有归属的像素, 设该像素为(x0, y0);
2. 以(x0, y0)为中心, 考虑(x0, y0)的4邻域像素(x, y)如果(x0, y0)满足生长准则, 将(x, y)与(x0, y0)合并(在同一区域内), 同时将(x, y)压入堆栈;
3. 从堆栈中取出一个像素, 把它当作(x0, y0)返回到步骤2;
4. 当堆栈为空时!返回到步骤1;
5. 重复步骤1 - 4直到图像中的每个点都有归属时。生长结束。
"""
import numpy as np
import cv2 as cv


def show_image(image):
    cv.namedWindow('input image', cv.WINDOW_NORMAL)
    cv.imshow('input image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


def get_gray_diff(image, current_point, temppoint):
    result = abs(int(image[current_point.x, current_point.y] - int(image[temppoint.x, temppoint.y])))
    return result


def select_connects(p):
    if p != 0:
        connects = [Point(-1, -1),
                    Point(0, -1),
                    Point(1, -1),
                    Point(1, 0),
                    Point(1, 1),
                    Point(0, 1),
                    Point(-1, 1),
                    Point(-1, 0)]
    else:
        connects = [Point(0, -1),
                    Point(1, 0),
                    Point(0, 1),
                    Point(-1, 0)]
    return connects


def region_grow(image, seeds, thresh, p=1):
    height, weight = image.shape
    seed_mark = np.zeros(image.shape)
    seed_list = []
    for seed in seeds:
        seed_list.append(seed)
    label = 1
    connects = select_connects(p)
    while len(seed_list) > 0:
        current_point = seed_list.pop(0)

        seed_mark[current_point.x,current_point.y] = label

        for i in range(8):
            tempX = current_point.x + connects[i].x
            tempY = current_point.y + connects[i].y

            if tempX < 0 or tempY < 0 or tempX >= height or tempY >= weight:
                continue

            gray_diff = get_gray_diff(image, current_point, Point(tempX, tempY))

            if gray_diff < thresh and seed_mark[tempX, tempY] == 0:
                seed_mark[tempX, tempY] = label
                seed_list.append(Point(tempX, tempY))
    return seed_mark


if __name__ == '__main__':
    image = cv.imread('../dataset/lena.png', 0)
    seeds = [Point(10, 10), Point(82, 150), Point(20, 300)]
    binary_image = region_grow(image, seeds, 10)
    show_image(binary_image)

























