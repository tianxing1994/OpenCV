"""
参考链接:
https://blog.csdn.net/huangwumanyan/article/details/82526873

快递面单识别:
快递面单有很多种形式的, 如果只是对单一的形式进行识别, 那在目标图片和面单模板之间, 做 SIFT 特征提取, 然后 flann 单应性匹配.
再切割指定位置的内容进行识别, 就好了.

但如果是需要针对多种形式的面单. 我的想法是先在图片中搜索如:
"收件人", "寄件人" 等标志性文字, 再分析以获取其内容的位置. 这个比较难吧.
"""
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


if __name__ == '__main__':
    image_path = '../dataset/data/express_paper/express_paper_1.jpg'

    image = cv.imread(image_path)

    show_image(image)
