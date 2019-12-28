from collections import Iterable
import numpy as np
import cv2 as cv


def psf2otf(psf, to_shape):
    h, w = psf.shape
    pad_size = (to_shape[0] - h, to_shape[1] - w)

    top = pad_size[0] // 2
    bottom = pad_size[0] - top
    left = pad_size[1] // 2
    right = pad_size[1] - left
    otf = cv.copyMakeBorder(src=psf, top=top, bottom=bottom, left=left, right=right, borderType=cv.BORDER_CONSTANT, value=0)
    np.roll(a=otf, shift=(pad_size[0] // 2, pad_size[1] // 2), axis=(0, 1))
    otf = np.fft.fft2(otf)
    ret = np.real(otf)
    return ret


if __name__ == '__main__':
    nd = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    result = psf2otf(nd, to_shape=(3, 3))
    print(result)
