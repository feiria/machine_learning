import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

rnd = np.random.RandomState(38)
x = rnd.uniform(-5, 5, size=50)
y_no_noise = (np.cos(6 * x) + x)
X = x.reshape(-1, 1)
y = (y_no_noise + rnd.normal(size=len(x))) / 2
line = np.linspace(-5, 5, 1000, endpoint=False).reshape(-1, 1)

poly = PolynomialFeatures(degree=20, include_bias=False)
X_poly = poly.fit_transform(X)
LNR_poly = LinearRegression().fit(X_poly, y)
line_poly = poly.transform(line)

plt.plot(line, LNR_poly.predict(line_poly), label='Linear Regressor')
plt.xlim(np.min(X) - 0.5, np.max(X) + 0.5)
plt.ylim(np.min(y) - 0.5, np.max(y) + 0.5)
plt.plot(X, y, 'o', c='r')
plt.legend(loc='lower right')
plt.show()
