import cv2 as cv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC, SVR

image_path = r'C:\Users\tianx\PycharmProjects\opencv\dataset\ps_image\image.jpg'
image = cv.imread(image_path)
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image = np.array(image / 255, dtype=np.int)

x = np.array(np.linspace(0, 100, 20, endpoint=False), np.int)
y = np.array(np.linspace(0, 100, 20, endpoint=False), np.int)

X, Y = np.meshgrid(x, y)
x = np.array(X, dtype=np.int)
y = np.array(Y, dtype=np.int)

data = np.stack([x, y], axis=2).reshape(-1, 2)
target = image[x, y].reshape(-1)

print(data.shape, target.shape)

# clf = SVC(kernel="poly")
# clf = SVC(kernel="rbf")
# clf = DecisionTreeClassifier()
clf = GaussianNB()
# clf = MultinomialNB()
# clf = BernoulliNB()

clf.fit(data, target)

x = np.array(np.linspace(0, 100, 100, endpoint=False), np.int)
y = np.array(np.linspace(0, 100, 100, endpoint=False), np.int)
X, Y = np.meshgrid(x, y)
x = np.array(X, dtype=np.int)
y = np.array(Y, dtype=np.int)

x_test = np.stack([x, y], axis=2).reshape(-1, 2)
predict = clf.predict(x_test)
result = np.ones(shape=(100, 100), dtype=np.uint8)
result[x_test[:, 0], x_test[:, 1]] = predict * 255

cv.imshow('result', result)
cv.imwrite(r'C:\Users\tianx\Desktop\result.jpg', result)
cv.waitKey(0)
cv.destroyAllWindows()
