import numpy as np
from sklearn.decomposition import PCA


X = np.array([[2, 4, 5, 1],
              [7, 5, 2, 4],
              [8, 5, 4, 2],
              [4, 3, 7, 9],
              [1, 2, 3, 1]])

pca = PCA(n_components=2)
result_pca = pca.fit_transform(X)
# print(result_pca)










