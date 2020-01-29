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

bins = np.linspace(-5, 5, 11)
target_bin = np.digitize(X, bins=bins)

onehot = OneHotEncoder(sparse=False)
onehot.fit(target_bin)
X_in_bin = onehot.transform(target_bin)

line = np.linspace(-5, 5, 1000, endpoint=False).reshape(-1, 1)
new_line = onehot.transform(np.digitize(line, bins=bins))

X_multi = np.hstack([X_in_bin, X * X_in_bin])
line_multi = np.hstack([new_line, line * new_line])
mlpr_multi = MLPRegressor().fit(X_multi, y)

plt.plot(line, mlpr_multi.predict(line_multi), label='MLP')
for vline in bins:
    plt.plot([vline, vline], [-5, 5], ':', c='gray')

plt.plot(X, y, 'o', c='r')
plt.legend(loc='lower right')
plt.show()
