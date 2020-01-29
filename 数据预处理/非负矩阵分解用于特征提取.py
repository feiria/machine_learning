from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

faces = fetch_lfw_people(min_faces_per_person=20, resize=0.8)
X_train, X_test, y_train, y_test = train_test_split(faces.data/255, faces.target, random_state=62)
nmf = NMF(n_components=105, random_state=62).fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)
print(X_train_nmf.shape)
mlp = MLPClassifier(hidden_layer_sizes=[100, 100], random_state=62, max_iter=400)
mlp.fit(X_train_nmf, y_train)
print('score:{:.2f}'.format(mlp.score(X_test_nmf, y_test)))
