import cv2 as cv
import numpy as np


# win_size = (32, 32)
# win_stride = (16, 16)
# block_size = (16, 16)
# block_stride = (16, 16)
# cell_size = (16, 16)
# nbins = 9

win_size = (32, 32)
win_stride = (4, 4)
block_size = (16, 16)
block_stride = (4, 4)
cell_size = (4, 4)
nbins = 9

args = (win_size, block_size, block_stride, cell_size, nbins)
hog = cv.HOGDescriptor(*args)


def get_hog_dataset(target_dataset, backgrounds):
    data = []
    target = []

    roi_list, label_list, channel_list = target_dataset
    for roi, label, channel in zip(roi_list, label_list, channel_list):
        h, w = roi.shape
        h_ret = int((h - win_size[1]) // win_stride[1] + 1)
        w_ret = int((w - win_size[0]) // win_stride[0] + 1)
        descriptors = hog.compute(roi, winStride=win_stride, padding=(0, 0))
        descriptors = descriptors.reshape(h_ret * w_ret, -1)
        data.append(descriptors)
        target.append(np.array([label]))

    for background in backgrounds:
        h, w = background.shape
        h_ret = int((h - win_size[1]) // win_stride[1] + 1)
        w_ret = int((w - win_size[0]) // win_stride[0] + 1)
        n = h_ret * w_ret
        descriptors = hog.compute(background, winStride=win_stride, padding=(0, 0))
        descriptors = descriptors.reshape(n, -1)
        data.append(descriptors)
        target.append(np.zeros(shape=(n,)))

    data = np.concatenate(data).astype(np.float32)
    target = np.concatenate(target).astype(np.int)
    return data, target


def custom_hog_detect(image, hog, svm):
    """OpenCV 的 hog.detect, hog.detectMultiScale 调用效果不好. 自己实现了一种方法
    由于只在 2 分类情况下 svm.predict() 参数 returnDFVal=True 时, 其返回值才是向量到分界平面的距离.
    所以对于重叠的 ROI, 先计算每个 ROI 被重叠的面积之和作为其 score. 然后进行非极大值抑制.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_h, image_w = gray.shape
    h_ret = int((image_h - win_size[1]) // win_stride[1] + 1)
    w_ret = int((image_w - win_size[0]) // win_stride[0] + 1)
    descriptors = hog.compute(gray, winStride=win_stride, padding=(0, 0))
    descriptors = descriptors.reshape(h_ret * w_ret, -1)
    _, target_ = svm.predict(descriptors)

    bounding_box_1 = list()
    idx_1, _ = np.where(target_ == 1)
    for index in idx_1:
        x1 = int((index % w_ret + 1) * win_stride[0])
        y1 = int((index // w_ret + 1) * win_stride[1])
        x2 = int(x1 + win_size[0])
        y2 = int(y1 + win_size[1])
        bounding_box_1.append([x1, y1, x2, y2])

    bounding_box_2 = list()
    idx_2, _ = np.where(target_ == 2)
    for index in idx_2:
        x1 = int((index % w_ret + 1) * win_stride[0])
        y1 = int((index // w_ret + 1) * win_stride[1])
        x2 = int(x1 + win_size[0])
        y2 = int(y1 + win_size[1])
        bounding_box_2.append([x1, y1, x2, y2])

    # 去除重叠的 ROI.
    bounding_box_1 = bounding_box_drop_overlapped(np.array(bounding_box_1))
    bounding_box_2 = bounding_box_drop_overlapped(np.array(bounding_box_2))
    return bounding_box_1, bounding_box_2


def bounding_box_drop_overlapped(bounding_boxes, thresh=0.3):
    if isinstance(bounding_boxes, list):
        bounding_boxes = np.array(bounding_boxes)
    h, w = bounding_boxes.shape

    x1 = bounding_boxes[:, 0]
    y1 = bounding_boxes[:, 1]
    x2 = bounding_boxes[:, 2]
    y2 = bounding_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    score = list()
    for i in range(h):
        xx1 = np.maximum(x1[i], x1[:])
        yy1 = np.maximum(y1[i], y1[:])
        xx2 = np.minimum(x2[i], x2[:])
        yy2 = np.minimum(y2[i], y2[:])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / (areas[i] + areas[:] - inter)
        inds = np.where(ovr >= thresh)[0]
        score.append(np.sum(inter[inds]))
    score = np.expand_dims(np.array(score, dtype=np.int), axis=1)
    bounding_boxes_with_score = np.hstack((bounding_boxes, score))

    bounding_boxes_drop_overlaped = non_maximum_suppression(bounding_boxes_with_score)
    ret = bounding_boxes_drop_overlaped[:, :4]
    return ret


def non_maximum_suppression(bounding_boxes, thresh=0.3):
    x1 = bounding_boxes[:, 0]
    y1 = bounding_boxes[:, 1]
    x2 = bounding_boxes[:, 2]
    y2 = bounding_boxes[:, 3]
    scores = bounding_boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    ret = bounding_boxes[keep]
    return ret


if __name__ == '__main__':
    bounding_boxes = np.array([[604, 248, 636, 280], [608, 248, 640, 280], [80, 252, 112, 284],
                               [600, 252, 632, 284], [604, 252, 636, 284], [608, 252, 640, 284],
                               [80, 256, 112, 288], [84, 256, 116, 288], [88, 256, 120, 288], [600, 256, 632, 288], [604, 256, 636, 288], [608, 256, 640, 288], [80, 260, 112, 292], [84, 260, 116, 292], [88, 260, 120, 292], [600, 260, 632, 292], [604, 260, 636, 292], [608, 260, 640, 292], [80, 264, 112, 296], [84, 264, 116, 296], [88, 264, 120, 296], [740, 316, 772, 348], [744, 316, 776, 348], [748, 316, 780, 348], [208, 320, 240, 352], [212, 320, 244, 352], [216, 320, 248, 352], [740, 320, 772, 352], [744, 320, 776, 352], [748, 320, 780, 352], [212, 324, 244, 356], [216, 324, 248, 356], [220, 324, 252, 356], [740, 324, 772, 356], [744, 324, 776, 356], [748, 324, 780, 356], [212, 328, 244, 360], [216, 328, 248, 360], [220, 328, 252, 360], [740, 328, 772, 360], [744, 328, 776, 360], [748, 328, 780, 360], [212, 332, 244, 364], [216, 332, 248, 364], [220, 332, 252, 364]])
    ret = bounding_box_drop_overlapped(bounding_boxes)
    print(ret)
