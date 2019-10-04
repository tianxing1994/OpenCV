import cv2 as cv


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    image_path = '../dataset/data/image_sample/inpaint_image.jpg'
    mask_path = '../dataset/data/image_sample/inpaint_mask.jpg'

    image = cv.imread(image_path)
    mask = cv.imread(mask_path, 0)

    show_image(image)
    show_image(mask)

    result = cv.inpaint(image, mask, 3, cv.INPAINT_TELEA)
    show_image(result)

    result = cv.inpaint(image, mask, 3, cv.INPAINT_NS)
    show_image(result)
    return


if __name__ == '__main__':
    demo1()
