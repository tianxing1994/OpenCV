"""
参考链接:
https://blog.csdn.net/weixin_33753845/article/details/87111017
https://github.com/yingshaoxo/MovingDetector
"""
import cv2
import sys
import numpy as np

try:
    import pyscreenshot as ImageGrab
    import pyautogui
    import time
except Exception as e:
    print(e)


class MovingDetector():
    def __init__(self):
        pass

    def camera_detect(self, device=0, func=None):
        self.camera = cv2.VideoCapture(device)
        self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.camera.isOpened():
            print("Could not open camera")
            sys.exit()

        history = 10  # 20    # 训练帧数

        bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # 背景减除器，设置阴影检测
        # bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
        bs.setHistory(history)

        frames = 0

        while True:
            res, frame = self.camera.read()

            if not res:
                break

            fg_mask = bs.apply(frame)  # 获取 foreground mask

            if frames < history:
                frames += 1
                continue

            # 对原始帧进行膨胀去噪
            th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
            th = cv2.erode(th, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
            dilated = cv2.dilate(th, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
            # 获取所有检测框
            contours, hier = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 0:
                # find the biggest area
                # c = min(contours, key=cv2.contourArea)
                c = max(contours, key=cv2.contourArea)
                # shrink area
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                # get center point
                x = int(sum([point[0] for point in box]) / 4)
                y = int(sum([point[1] for point in box]) / 4)
                # convert all coordinates floating point values to int
                box = np.int0(box)
                # draw a red 'nghien' rectangle
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                # run comstomer function
                if func != None:
                    func({"info": {"width": self.width, "height": self.height},
                          "center_point": (x, y), "box": box})
                else:
                    print("the center point is: ({x}, {y})".format(x=x, y=y))

            cv2.imshow("detection", frame)
            cv2.imshow("back", dilated)

            if cv2.waitKey(110) & 0xff == 27:
                break

    def screen_detect(self, record_box_size=600):
        record_box_size = record_box_size // 2

        history = 10  # 训练帧数

        bs = cv2.createBackgroundSubtractorKNN(
            detectShadows=True)  # 背景减除器，设置阴影检测
        bs.setHistory(history)

        frames = 0

        while True:
            pos = pyautogui.position()
            mouse_x = pos[0]
            mouse_y = pos[1]
            left_top_x = mouse_x - record_box_size
            left_top_y = mouse_y - record_box_size
            right_bottom_x = mouse_x + record_box_size
            right_bottom_y = mouse_y + record_box_size

            frame = np.array(ImageGrab.grab(
                bbox=(left_top_x, left_top_y, right_bottom_x, right_bottom_y)))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            fg_mask = bs.apply(frame)  # 获取 foreground mask

            if frames < history:
                frames += 1
                continue

            # 对原始帧进行膨胀去噪
            th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
            th = cv2.erode(th, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
            dilated = cv2.dilate(th, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
            # 获取所有检测框
            image, contours, hier = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 0:
                # find the biggest area
                c = max(contours, key=cv2.contourArea)
                # shrink area
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)

                # get center point
                x = int(sum([point[0] for point in box]) / 4)
                y = int(sum([point[1] for point in box]) / 4)
                x = left_top_x + x
                y = left_top_y + y
                print("center point: ({x}, {y})".format(x=x, y=y))
                # pyautogui.moveTo(x, y)

                # convert all coordinates floating point values to int
                box = np.int0(box)
                # draw a red 'nghien' rectangle
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

            cv2.imshow("detection", frame)
            # cv2.imshow("back", dilated)

            if cv2.waitKey(110) & 0xff == 27:
                break


def func(return_values):
    camera_info = return_values["info"]
    width = camera_info["width"]
    half_width = width / 2

    middle_width = 0.3 * width

    first_line = half_width - middle_width / 2
    second_line = half_width + middle_width / 2

    x = return_values["center_point"][0]
    if x < first_line:
        print("left")
    elif first_line < x < second_line:
        print("middle")
    elif x > second_line:
        print("right")


if __name__ == '__main__':
    detector = MovingDetector()
    detector.camera_detect(device=0, func=func)
