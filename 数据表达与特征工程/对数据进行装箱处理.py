import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder

rnd = np.random.RandomState(38)
x = rnd.uniform(-5, 5, size=50)
y_no_noise = (np.cos(6 * x) + x)
X = x.reshape(-1, 1)
y = (y_no_noise + rnd.normal(size=len(x))) / 2
plt.plot(X, y, 'o', c='r')
plt.show()

line = np.linspace(-5, 5, 1000, endpoint=False).reshape(-1, 1)
mlpr = MLPRegressor().fit(X, y)
knr = KNeighborsRegressor().fit(X, y)
plt.plot(line, mlpr.predict(line), label='MLP')
plt.plot(line, knr.predict(line), label='KNN')
plt.plot(X, y, 'o', c='r')
plt.legend(loc='best')
plt.show()

bins = np.linspace(-5, 5, 11)
target_bin = np.digitize(X, bins=bins)
print(bins)

onehot = OneHotEncoder(sparse=False)
onehot.fit(target_bin)
X_in_bin = onehot.transform(target_bin)
new_line = onehot.transform(np.digitize(line, bins=bins))

new_mlpr = MLPRegressor().fit(X_in_bin, y)
new_knr = KNeighborsRegressor().fit(X_in_bin, y)
plt.plot(line, new_mlpr.predict(new_line), label='New MLP')
plt.plot(line, new_knr.predict(new_line), label='New KNN')
plt.plot(X, y, 'o', c='r')
plt.legend(loc='best')
plt.show()
