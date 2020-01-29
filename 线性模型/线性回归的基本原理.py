# 导入数据集拆分工具
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=2, n_informative=2, random_state=38)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
lr = LinearRegression()
lr.fit(X_train, y_train)
print('\n代码运行结果： ')
print('lf.coef_:{}'.format(lr.coef_[:]))
print('lr.intercept_:{}'.format(lr.intercept_))