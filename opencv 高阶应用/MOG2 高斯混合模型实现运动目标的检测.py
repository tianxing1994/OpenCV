"""
https://blog.csdn.net/mago2015/article/details/85332363
"""
import cv2 as cv
import numpy as np


def show_image(image):
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


cap = cv.VideoCapture(r"C:\Users\Administrator\PycharmProjects\OpenCV\dataset\data\768x576.avi")

size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

# 定义高斯混合模型对象.
mog = cv.createBackgroundSubtractorMOG2(history=50, detectShadows=True)

# fourcc1 = cv.VideoWriter_fourcc(*'XVID')
# fourcc2 = cv.VideoWriter_fourcc(*'XVID')
# out_detect = cv.VideoWriter('output_detect.avi', fourcc1, 20.0, size)
# out_bg = cv.VideoWriter('output_bg.avi', fourcc1, 20.0, size)

i = 0
while True:
    ret, frame = cap.read()

    fgmask = mog.apply(frame)
    gray = fgmask.copy()

    gray, cnts, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv.contourArea(c) < 900:
            continue
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 255), 2)

    cv.imshow("contours", frame)
    cv.imshow("fgmask", gray)

    # image = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    # out_detect.write(frame)
    # out_bg.write(image)

    i += 1

    if cv.waitKey(int(1000 / 12)) & 0xff == ord("q"):
        break


cap.release()
# out_detect.release()
# out_bg.release()
cv.destroyAllWindows()
