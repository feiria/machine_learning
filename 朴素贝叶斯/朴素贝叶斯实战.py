# 导入威斯康星乳腺肿瘤数据集
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


cancer = load_breast_cancer()

print('keys')
print(cancer.keys())

print('肿瘤的分类', cancer['target_names'])
print('肿瘤的特征', cancer['feature_names'])

X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=38)
print('训练集数据形态', X_train.shape)
print('测试集数据形态', X_test.shape)

# 使用高斯朴素贝叶斯拟合数据
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('\n代码运行结果： ')
print('训练集数据得分：{:.3f}'.format(gnb.score(X_train, y_train)))
print('测试及数据得分：{:.3f}'.format(gnb.score(X_test, y_test)))

print('模型预测的分类是:{}'.format(gnb.predict([X[312]])))
print('模型正确的分类是:{}'.format(y[312]))