# 导入DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


blobs = make_blobs(random_state=1, centers=1)
X_blobs = blobs[0]
db = DBSCAN(eps=2, min_samples=20)
clusters = db.fit_predict(X_blobs)
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=clusters, cmap=plt.cm.cool, s=60, edgecolors='k')
plt.xlabel('feature 0')
plt.ylabel('feature 1')
plt.show()