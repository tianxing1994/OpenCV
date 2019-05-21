import cv2 as cv

image_path = 'C:/Users/tianx/PycharmProjects/opencv/dataset/contours.png'
src = cv.imread(image_path)

# 创建一个名为 input image 的窗口
cv.namedWindow('input image', cv.WINDOW_AUTOSIZE)
# 在名为 input image 的窗口中显示图片
cv.imshow('input image', src)
# waitKey(0) 则等待用户的操作后释放窗口内存, 如果不为 0, 则等待指定的毫秒数后释放内存.
cv.waitKey(4000)
cv.destroyAllWindows()
print("Hi, Python!")