from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_diabetes().data, load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)

lasso = Lasso().fit(X_train, y_train)
print('\n代码运行结果： ')
print('训练集数据得分：{:.2f}'.format(lasso.score(X_train, y_train)))
print('测试及数据得分：{:.2f}'.format(lasso.score(X_test, y_test)))
print('使用的特征数：{}'.format(np.sum(lasso.coef_ != 0)))

lasso01 = Lasso(alpha=0.1, max_iter=100000).fit(X_train, y_train)
print('\n代码运行结果： ')
print('训练集数据得分：{:.2f}'.format(lasso01.score(X_train, y_train)))
print('测试及数据得分：{:.2f}'.format(lasso01.score(X_test, y_test)))
print('使用的特征数：{}'.format(np.sum(lasso01.coef_ != 0)))

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print('\n代码运行结果： ')
print('训练集数据得分：{:.2f}'.format(lasso00001.score(X_train, y_train)))
print('测试及数据得分：{:.2f}'.format(lasso00001.score(X_test, y_test)))
print('使用的特征数：{}'.format(np.sum(lasso00001.coef_ != 0)))