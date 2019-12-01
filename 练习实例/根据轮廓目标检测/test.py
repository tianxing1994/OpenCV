import os
import sys
current_path = os.getcwd()
sys.path.insert(0, current_path)

import cv2 as cv
import numpy as np

from util import show_image, calc_canny_edge


if __name__ == '__main__':

    # 加载加, 进度条, 检测出直线, 必须要有平行的两条, 且, 两条需长度相当, 距离相近.
    # image_dir = '../../dataset/local_dataset/hepingjingying'
    image_dir = '../../dataset/local_dataset/chuanyuehuoxian'
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            image_path = os.path.join(root, file)

            image = cv.imread(image_path)
            edge = calc_canny_edge(image)
            show_image(edge)

