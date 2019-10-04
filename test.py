"""
参考链接:
https://blog.csdn.net/foreverhot1019/article/details/78793816
"""
import numpy as np
import cv2 as cv


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def sigmentation(project, projection_noise_reduction=True):
    l = len(project)
    project_mean = np.mean(project[project != 0])
    project_std = np.std(project[project != 0])
    if projection_noise_reduction:
        project_ = np.array(np.where(project > (project_mean - 1.25 * project_std), 1, 0))
    else:
        project_ = np.array(np.where(project > 0, 1, 0))

    segmentation_l = list()
    last_label = 0
    start_i = -1
    end_i = -1
    length = 0
    for i in range(0, l):
        this_label = project_[i]
        if this_label != last_label and this_label > 0:
            start_i = i
        elif this_label != last_label and this_label == 0:
            end_i = i
            length = end_i - start_i
            segmentation_l.append((start_i, end_i, length))
            start_i, end_i, length = -1, -1, 0
        else:
            pass
        last_label = this_label
    segmentation_l = np.array(segmentation_l)
    return segmentation_l


def get_segment_data(binary, anchor=(0, 0)):
    h_project = np.sum(binary, axis=1)
    h_segments = sigmentation(h_project, projection_noise_reduction=False)
    w_segments_list = list()
    for h_segment in h_segments:
        s, e, _ = h_segment
        w_project = np.sum(binary[s: e, :], axis=0)
        w_segments = sigmentation(w_project,
                                  projection_noise_reduction=False)
        w_segments_list.append(w_segments)

    result = list()
    x_, y_ = anchor
    for i in range(len(h_segments)):
        y1, y2, _ = h_segments[i]
        for j in range(len(w_segments_list[i])):
            x1, x2, _ = w_segments_list[i][j]
            result.append((x1+x_, y1+y_, x2+x_, y2+y_))
    return result


def draw_line(image, bounding_list):
    for x1, y1, x2, y2 in bounding_list:
        cv.line(image, (x1, y1), (x1, y2), (0, 0, 255), 1)
        cv.line(image, (x1, y1), (x2, y1), (0, 0, 255), 1)
        cv.line(image, (x1, y2), (x2, y2), (0, 0, 255), 1)
        cv.line(image, (x2, y1), (x2, y2), (0, 0, 255), 1)
    return image


def significant_image(image):
    """著于傅里叶变换的, 显著性检测"""
    dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
    dft = np.fft.fftshift(dft)

    # 幅度谱, 相位谱
    gray_spectrum = np.log(cv.magnitude(dft[:, :, 0], dft[:, :, 1]))
    phase_spectrum = np.arctan2(dft[:, :, 1], dft[:, :, 0])

    gray_spectrum_mean = cv.blur(gray_spectrum, ksize=(3, 3))
    spectral_residual = np.exp(gray_spectrum - gray_spectrum_mean)

    # 余弦谱, 正弦谱
    cos_spectrum = np.expand_dims(np.cos(phase_spectrum) * spectral_residual, axis=2)
    sin_spectrum = np.expand_dims(np.sin(phase_spectrum) * spectral_residual, axis=2)

    new_dft = np.concatenate([cos_spectrum, sin_spectrum], axis=2)
    idft = cv.dft(new_dft, cv.DFT_INVERSE)
    idft_magnitude = cv.magnitude(idft[:, :, 0], idft[:, :, 1])
    idft_magnitude = np.array(idft_magnitude / idft_magnitude.max() * 255, dtype=np.uint8)
    return idft_magnitude


def significant_image_demo():
    # image_path = 'dataset/data/bank_card/card_ 1.png'
    # image_path = 'dataset/data/bank_card/card_ 2.png'
    # image_path = 'dataset/data/bank_card/card_ 3.png'
    # image_path = 'dataset/data/bank_card/card_ 4.png'
    image_path = 'dataset/data/bank_card/card_ 5.png'

    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    idft_magnitude = significant_image(gray)
    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 3))
    tophat = cv.morphologyEx(idft_magnitude, cv.MORPH_TOPHAT, rect_kernel)
    show_image(tophat)
    tophat_blur = cv.blur(tophat, ksize=(3, 3))

    _, binary = cv.threshold(tophat_blur, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    sq_kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 1))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, sq_kernel)
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, sq_kernel)
    show_image(binary)


    return


if __name__ == '__main__':
    significant_image_demo()
