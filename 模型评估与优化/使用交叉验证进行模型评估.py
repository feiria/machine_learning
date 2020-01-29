from sklearn.datasets import load_wine
# 导入交叉验证工具
from sklearn.model_selection import cross_val_score
# 导入随机拆分工具
from sklearn.model_selection import ShuffleSplit
# 导入LeaveOneOut
from sklearn.model_selection import LeaveOneOut
# 导入用于分类的支持向量机模型
from sklearn.svm import SVC

wine = load_wine()
# 设置SVC的核心函数为linear
svc = SVC(kernel='linear')
scores = cross_val_score(svc, wine.data, wine.target, cv=6)
print(scores)
print('{:.3f}'.format(scores.mean()))

# 随机拆分
shuffle_split = ShuffleSplit(test_size=.2, train_size=.7, n_splits=10)
scores = cross_val_score(svc, wine.data, wine.target, cv=shuffle_split)
print(scores)
print('{:.3f}'.format(scores.mean()))

# 挨个试试
cv = LeaveOneOut()
scores = cross_val_score(svc, wine.data, wine.target, cv=cv)
print('{:.3f}'.format(scores.mean()))