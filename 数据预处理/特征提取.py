from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

faces = fetch_lfw_people(min_faces_per_person=20, resize=0.8)
image_shape = faces.images[0].shape
fig, axes = plt.subplots(3, 4, figsize=(12, 9), subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(faces.target, faces.images, axes.ravel()):
    ax.imshow(image, cmap=plt.cm.gray)
    ax.set_title(faces.target_names[target])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(faces.data/255, faces.target, random_state=62)
mlp = MLPClassifier(hidden_layer_sizes=[100, 100], random_state=62, max_iter=400)
mlp.fit(X_train, y_train)
print(X_train.shape)
print('score:{:.2f}'.format(mlp.score(X_test, y_test)))

pca = PCA(whiten=True, n_components=0.9, random_state=62)
pca.fit(X_train)
X_train_whiten = pca.transform(X_train)
X_test_whiten = pca.transform(X_test)
mlp.fit(X_train_whiten, y_train)
print(X_train_whiten.shape)
print('score:{:.2f}'.format(mlp.score(X_test_whiten, y_test)))