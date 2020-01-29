# 导入高斯朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
# 导入画图工具
import matplotlib.pyplot as plt
# 导入数据集生成工具
from sklearn.datasets import make_blobs
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split
import numpy as np

# 生成样本数量为500， 分类数为5的数据集
X, y = make_blobs(n_samples=500, centers=5, random_state=8)
# 将数据集拆分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)

# 使用高斯朴素贝叶斯
gnb = GaussianNB()
gnb.fit(X_train, y_train)

print('\n代码运行结果： ')
print('训练集数据得分：{:.3f}'.format(gnb.score(X_train, y_train)))
print('测试及数据得分：{:.3f}'.format(gnb.score(X_test, y_test)))

# 限定横轴与纵轴的最大值
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
# 用不同的背景色表示不同的分类
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                     np.arange(y_min, y_max, .02))
z = gnb.predict(np.c_[(xx.ravel(), yy.ravel())]).reshape(xx.shape)
plt.pcolormesh(xx, yy, z, cmap=plt.cm.Spectral)
# 将训练集和测试集用散点图表示
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.cool, edgecolors='k')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.cool, marker='*', edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('Classifier GaussianNB')
plt.show()