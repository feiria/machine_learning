from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=38)
params = {
    'alpha': [0.01, 0.1, 1.0, 10.0],
    'max_iter': [100, 1000, 5000, 10000]
}
gird_search = GridSearchCV(Lasso(), params, cv=6)
gird_search.fit(X_train, y_train)
print(gird_search.score(X_test, y_test))
print(gird_search.best_params_)
print(gird_search.best_score_)
