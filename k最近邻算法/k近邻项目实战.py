from sklearn.datasets import load_wine
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split
# 导入KNN分类模型
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 从sklearn的datasets模块载入数据集
wine_dataset = load_wine()
print("红酒数据集中的键:\n{}".format(wine_dataset.keys()))
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])
# 数据    目标分类    目标分类名称  数据描述    特征变量名称
print("数据概况：{}".format(wine_dataset['data'].shape))
# 数据概况：(178, 13)
# 178个样本 13个特征变量
# 打印酒的数据集中的简短描述
# print(wine_dataset['DESCR'])

# 将数据集拆分为训练数据集和测试数据集
X_train, X_test, y_train, y_test = train_test_split(
    wine_dataset['data'], wine_dataset['target'], random_state=0)

# 打印训练数据集中特征向量的形态
print('X_train shape:{}'.format(X_train.shape))
# 打印测试数据集中特征向量的形态
print('X_test shape:{}'.format(X_test.shape))
# 打印训练数据集中目标的形态
print('y_train shape:{}'.format(y_train.shape))
# 打印训练数据集中目标的形态
print('y_test shape:{}'.format(y_test.shape))

# 使用k最近邻算法进行建模
# 指定模型的n_neighbors参数值为1
knn = KNeighborsClassifier(n_neighbors=1)
# 用模型对数据进行拟合
knn.fit(X_train, y_train)
print(knn)
print("\n测试数据得分：{:.2f}".format(knn.score(X_test, y_test)))

# 输入新的数据点
X_new = np.array([[13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820]])
# 使用.predict进行预测
prediction = knn.predict(X_new)
print("预测新红酒的分类是：{}".format(wine_dataset['target_names'][prediction]))
