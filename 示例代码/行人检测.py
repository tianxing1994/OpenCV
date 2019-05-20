import cv2 as cv
import numpy as np


def is_inside(o, i):
    ox, oy, ow, oh, = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih


def draw_person(image, person):
    x, y, w, h = person
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return image


def people_detector():
    image = cv.imread('../dataset/other/people.jpg')
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    found, w = hog.detectMultiScale(image)

    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and is_inside(r, q):
                break
        else:
            found_filtered.append(r)

    for person in found_filtered:
        image = draw_person(image, person)

    cv.imshow('people detection', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


if __name__ == '__main__':
    people_detector()