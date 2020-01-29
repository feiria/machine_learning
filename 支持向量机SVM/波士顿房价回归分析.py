from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

boston = load_boston()
print(boston.keys())
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
print(X_train.shape)
print(X_test.shape)

scaled = StandardScaler()
scaled.fit(X_train)
X_train_scaled = scaled.transform(X_train)
X_test_scaled = scaled.transform(X_test)


for kernel in ['linear', 'rbf']:
    svr = SVR(kernel=kernel, C=100, gamma=0.1)
    svr.fit(X_train_scaled, y_train)
    print(kernel, 'train: {:.3f}'.format(svr.score(X_train_scaled, y_train)))
    print(kernel, 'test: {:.3f}'.format(svr.score(X_test_scaled, y_test)))