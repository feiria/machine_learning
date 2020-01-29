from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
# 导入岭回归
from sklearn.linear_model import Ridge
# 载入糖尿病数据集
X, y = load_diabetes().data, load_diabetes().target
# 将数据集拆分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
# 使用岭回归对数据进行拟合
ridge = Ridge().fit(X_train, y_train)

print('\n代码运行结果： ')
print('训练集数据得分：{:.2f}'.format(ridge.score(X_train, y_train)))
print('测试及数据得分：{:.2f}'.format(ridge.score(X_test, y_test)))

ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print('\n代码运行结果： ')
print('训练集数据得分：{:.2f}'.format(ridge10.score(X_train, y_train)))
print('测试及数据得分：{:.2f}'.format(ridge10.score(X_test, y_test)))

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print('\n代码运行结果： ')
print('训练集数据得分：{:.2f}'.format(ridge01.score(X_train, y_train)))
print('测试及数据得分：{:.2f}'.format(ridge01.score(X_test, y_test)))