# 导入随机森林模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
# 载入红酒数据集
wine = load_wine()
# 选择数据集的前两个特征
X = wine.data[:, :2]
y = wine.target
# 将数据集拆分成为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y)
# 设定随机森林有六颗树
forest = RandomForestClassifier(n_estimators=6, random_state=3)
# 使用模型拟合数据
forest.fit(X_train, y_train)

# 定义图像中分区的颜色和散点的颜色
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# 分别用样本的两个特征值创建图像的横轴和纵轴
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                     np.arange(y_min, y_max, .02))
Z = forest.predict(np.c_[xx.ravel(), yy.ravel()])

# 给每个分类中的样本分配不同的颜色
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# 用散点图把样本表示出来
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('RandomForest')
plt.show()