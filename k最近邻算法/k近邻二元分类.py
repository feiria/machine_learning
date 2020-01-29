# 导入数据集生成器
from sklearn.datasets import make_blobs
# 导入KNN分类器
from sklearn.neighbors import KNeighborsClassifier
# 导入画图工具
import matplotlib.pyplot as plt
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split
import numpy as np

# 生成样本数为200， 分类为2的数据集
data = make_blobs(n_samples=200, centers=2, random_state=8)
X, y = data
# print(X)
# X为点的坐标 X[:, 0]为全部点的x坐标, X[:, 1]为全部点的y坐标, y为标签
# 将生成的数据可视化
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.spring, edgecolor='k')
plt.show()

# 拟合数据

clf = KNeighborsClassifier()
clf.fit(X, y)

# 下面的代码用于画图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                     np.arange(y_min, y_max, .02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.spring, edgecolor='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classifier:KNN")
# 把新的数据点表示出来
plt.scatter(6.75, 4.82, marker='*', c='blue', s=200)
plt.show()
