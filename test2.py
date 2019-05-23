import cv2


image_path = r'C:\Users\Administrator\PycharmProjects\openCV\dataset\example.png'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hod = cv2.HOGDescriptor()

hod.detect(gray)


result = hod.compute(image)
print(result)
print(result.shape)
print(result.sum())

