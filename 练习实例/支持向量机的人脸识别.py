"""
参考链接
https://blog.csdn.net/qq_41302130/article/details/82937698
"""
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people()

print(faces.target_names)
print(faces.images.shape)


