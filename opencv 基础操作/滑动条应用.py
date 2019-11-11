import cv2 as cv
import numpy as np


def nothing(x):
    pass


# Create a black image, a window
img = np.zeros((300, 512, 3), np.uint8)
cv.namedWindow('image', cv.WINDOW_NORMAL)

# create trackbars for color change
cv.createTrackbar('R', 'image', 0, 255, nothing)
cv.createTrackbar('G', 'image', 0, 255, nothing)
cv.createTrackbar('B', 'image', 0, 255, nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv.createTrackbar(switch, 'image', 0, 1, nothing)
# 只有当 转换按钮 指向 ON 时 滑动条的滑动才有用，否则窗户 都是黑的。

while True:

    # get current positions of four trackbars
    r = cv.getTrackbarPos('R', 'image')
    g = cv.getTrackbarPos('G', 'image')
    b = cv.getTrackbarPos('B', 'image')
    s = cv.getTrackbarPos(switch, 'image')  # 另外一个重要应用就是用作转换按钮

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, g, r]

    cv.imshow('image', img)
    k = cv.waitKey(1)  # & 0xFF
    if k == ord("q"):
        break

cv.destroyAllWindows()