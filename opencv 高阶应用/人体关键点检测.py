# -*-coding: utf-8 -*-
"""
人体姿态估计.

参考链接:
https://github.com/PanJinquan/opencv-learning-tutorials/tree/master/opencv_dnn_pro/openpose-opencv

模型下载地址:
https://pan.baidu.com/s/17Ci0elj1cqSU_QV_tPzkWw
"""
import cv2 as cv
import glob


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]


def detect_key_point(model_path, image_path, inWidth=368, inHeight=368, threshhold=0.2):
    net = cv.dnn.readNetFromTensorflow(model_path)
    frame = cv.imread(image_path)
    frameHeight, frameWidth = frame.shape[:2]
    scalefactor = 2.0
    net.setInput(cv.dnn.blobFromImage(frame, scalefactor, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    assert (len(BODY_PARTS) == out.shape[1])
    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]
        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > threshhold else None)
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    show_image(frame)
    return


def detect_image_list_key_point(image_dir, out_dir, inWidth=480, inHeight=480, threshhold=0.3):
    image_list = glob.glob(image_dir)
    for image_path in image_list:
        detect_key_point(image_path, out_dir, inWidth, inHeight, threshhold)


def demo1():
    model_path = "../dataset/pose_estimation/model/graph_opt.pb"
    image_path = "../dataset/pose_estimation/test.jpg"
    detect_key_point(model_path, image_path, inWidth=368, inHeight=368, threshhold=0.05)
    return


def demo2():
    """
    预测人体关键点并描绘到图中.
    out = net.forward() 的返回值 out 形状为 n, c, h, w.
    n 代表输入数据的第 n 个需预测图片.
    c 为 57, 表示人体的 57 个关键点.
    h, w 是 46, 表示图片上的 46*46 个锚点.
    矩阵中的每一个值表示该锚点为对应关键点的可能性.
    """
    model_path = "../dataset/pose_estimation/model/graph_opt.pb"
    image_path = "../dataset/pose_estimation/test.jpg"
    net = cv.dnn.readNetFromTensorflow(model_path)
    frame = cv.imread(image_path)
    frameHeight, frameWidth = frame.shape[:2]

    scalefactor = 2.0
    net.setInput(cv.dnn.blobFromImage(frame, scalefactor, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))

    out = net.forward()
    n, c, h, w = out.shape
    print(n, c, h, w)
    for i in range(c):
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(out[0, i, :, :])
        x = int((frameWidth * max_loc[0]) / w)
        y = int((frameHeight * max_loc[1]) / h)
        frame = cv.circle(img=frame,
                          center=(x, y),
                          radius=2,
                          color=(0, 0, 255),
                          thickness=-1,
                          lineType=cv.LINE_AA)
    show_image(frame)
    return


if __name__ == "__main__":
    demo2()
