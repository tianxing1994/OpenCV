"""
相关函数:
cv2.matchTemplate
cv2.rectangle
"""
import cv2 as cv
import numpy as np


def template_demo():
    target = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/image0.JPG")
    template = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/template.jpg")
    # cv.imshow("target image", target)
    # cv.imshow("template image", template)
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    th, tw = template.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(target, template, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0]+tw, tl[1]+th)
        cv.rectangle(target, tl, br, (0, 0, 255), 2)
        cv.imshow("match-" + np.str(md), target)
        # cv.imshow("match-" + np.str(md), result)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return


if __name__ == '__main__':
    template_demo()