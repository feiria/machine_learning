import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=38)
best_score = 0
best_parameters = {}
for alpha in [0.01, 0.1, 1.0, 10.0]:
    for max_iter in [100, 1000, 5000, 10000]:
        lasso = Lasso(alpha=alpha, max_iter=max_iter)
        lasso.fit(X_train, y_train)
        score = cross_val_score(lasso, X_train, y_train, cv=6)
        score = np.mean(score)
        if best_score < score:
            best_score = score
            best_parameters = {
                'alpha': alpha,
                'max_iter': max_iter
            }
print(best_score)
print(best_parameters)
lasso = Lasso(alpha=best_parameters.get('alpha'), max_iter=best_parameters.get('max_iter')).fit(X_train, y_train)
print(lasso.score(X_test, y_test))
