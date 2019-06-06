import numpy as np


nd = np.array([[0, 0, 3],
               [1, 0, 3],
               [1, 0, 0]])

result = np.logical_or.reduce(nd, axis=None, keepdims=True)

print(result)