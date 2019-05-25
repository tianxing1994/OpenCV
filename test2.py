import cv2 as cv


def show_image(image):
    cv.namedWindow('input image', cv.WINDOW_NORMAL)
    cv.imshow('input image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return

image_path = r'C:/Users/Administrator/PycharmProjects/openCV/dataset2/carData/TrainImages/neg-0.pgm'
image = cv.imread(image_path)

show_image(image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

hog = cv.HOGDescriptor()

result = hog.detect(gray)
# result = hog.compute(image)



print(result)


