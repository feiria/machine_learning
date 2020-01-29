import numpy as np
from scipy import sparse

matrix = np.eye(6)
sparse_matrix = sparse.csr_matrix(matrix)

print("对角矩阵：\n{}".format(matrix))
print("\nsparse存储的矩阵：\n{}".format(sparse_matrix))