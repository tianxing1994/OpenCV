import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


image = cv.imread('dataset/data/exercise_image/image_text_r.png', 0)
h, w = image.shape
center_array = np.array([np.power(-1, r+c) for r in range(h) for c in range(w)]).reshape(image.shape)

center_image = image * center_array
print(center_image)

dft = cv.dft(np.float32(center_image), flags=cv.DFT_COMPLEX_OUTPUT)
dft_magnitude = np.log(cv.magnitude(dft[:, :, 0], dft[:, :, 1]))
dft_magnitude = np.array(dft_magnitude / dft_magnitude.max() * 255, dtype=np.uint8)
show_image(dft_magnitude)
fshift = np.fft.fftshift(f)
