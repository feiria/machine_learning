import numpy
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer


X, y = make_blobs(n_samples=40, centers=2, random_state=50, cluster_std=2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.cool)
plt.show()

# 将所有数据的特征值转换为均值为0， 而方差为1的状态
X1 = StandardScaler().fit_transform(X)
plt.scatter(X1[:, 0], X1[:, 1], c=y, cmap=plt.cm.cool)
plt.show()

# 将所有数据压进一个1*1的格子中
X2 = MinMaxScaler().fit_transform(X)
plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap=plt.cm.cool)
plt.show()

# 使用中位数和四分数， 会去掉异常值
X3 = RobustScaler().fit_transform(X)
plt.scatter(X3[:, 0], X3[:, 1], c=y, cmap=plt.cm.cool)
plt.show()

# 将所有样本的特征向量转化为欧几里得距离为1，也就是说， 他把数据的分布变成一个半径为1的⚪
X4 = Normalizer().fit_transform(X)
plt.scatter(X4[:, 0], X4[:, 1], c=y, cmap=plt.cm.cool)
plt.show()