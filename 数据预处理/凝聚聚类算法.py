from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, ward


blobs = make_blobs(random_state=1, centers=1)
X_blobs = blobs[0]
linkage = ward(X_blobs)
dendrogram(linkage)
ax = plt.gca()
plt.xlabel('Sample index')
plt.ylabel('Cluster distance')
plt.show()