from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

scaler = StandardScaler()
wine = load_wine()
X = wine.data
y = wine.target
x_scaled = scaler.fit_transform(X)
print(x_scaled.shape)

pca = PCA(n_components=2)
pca.fit(x_scaled)
x_pca = pca.fit_transform(x_scaled)
print(x_pca.shape)

x0 = x_pca[wine.target == 0]
x1 = x_pca[wine.target == 1]
x2 = x_pca[wine.target == 2]

plt.scatter(x0[:, 0], x0[:, 1], c='b', s=60, edgecolors='k')
plt.scatter(x0[:, 0], x0[:, 1], c='g', s=60, edgecolors='k')
plt.scatter(x0[:, 0], x0[:, 1], c='r', s=60, edgecolors='k')

plt.legend(wine.target_names, loc='best')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.show()