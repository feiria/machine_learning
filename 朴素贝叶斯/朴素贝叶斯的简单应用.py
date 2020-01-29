import numpy as np
# 导入贝努利贝叶斯
from sklearn.naive_bayes import BernoulliNB

X = np.array([
    [0, 1, 0, 1],
    [1, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 0, 1]
])
y = np.array([0, 1, 1, 0, 1, 0, 0])
counts = {}
for label in np.unique(y):
    print(label)
    print(y == label)
    print(X[y == label])
    counts[label] = X[y == label].sum(axis=0)
print(counts)

# 使用贝努利贝叶斯拟合数据
clf = BernoulliNB()
clf.fit(X, y)
Next_Day = [[0, 0, 1, 0]]
pre = clf.predict(Next_Day)
if pre == [1]:
    print('rain')
else:
    print('sun')
print(clf.predict_proba(Next_Day))