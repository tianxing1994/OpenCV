import numpy as np
import cv2 as cv


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


ret = cv.getGaborKernel(ksize=(60, 60), sigma=9, theta=np.pi/3, lambd=8, gamma=1, psi=np.pi)

min_val, max_val, _, _ = cv.minMaxLoc(ret)

ret += max_val
ret /= (2*max_val)

show_image(ret)
