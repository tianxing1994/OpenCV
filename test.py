import cv2 as cv

detect = cv.xfeatures2d.SIFT_create()

image_path = 'dataset/data/image_sample/lena.png'
image = cv.imread(image_path)

keypoints= detect.detect(image)

print(keypoints)

print(keypoints[0].pt)