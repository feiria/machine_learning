import numpy as np
import matplotlib.pyplot as plt
# 导入线性回归模型
from sklearn.linear_model import LinearRegression

# 输入3个点的横坐标
X = [[1], [4], [3]]
# 输入3个点的从坐标
y = [3, 5, 3]
# 用线性模型拟合这三个点
lr = LinearRegression().fit(X, y)
z = np.linspace(0, 5, 20)
plt.scatter(X, y, s=80)
plt.plot(z, lr.predict(z.reshape(-1, 1)), c='k')
plt.title('Straight Line')
plt.show()