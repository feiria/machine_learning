import pandas as pd
# 用pandas打开csv文件
from sklearn import tree
from IPython.core.display import display
from sklearn.model_selection import train_test_split

data = pd.read_csv('adult.csv', header=None, index_col=False,
                   names=['年龄', '单位性质', '权重', '学历', '受教育时长',
                          '婚姻状况', '职业', '家庭情况', '种族', '性别',
                          '资产所得', '资产损失', '周工作时长', '原籍', '收入'])
# 为了方便展示， 我们选区区其中一部分数据
data_lite = data[['年龄', '单位性质', '学历', '性别', '周工作时长', '职业', '收入']]
display(data_lite.head())

# 使用get_dummies将文本数据妆化为数值
data_dummies = pd.get_dummies(data_lite)
# 对比样本原始特征和虚拟变量特征
print('样本原始特征：\n', list(data_lite.columns), '\n')
print('虚拟变量特征：\n', list(data_dummies.columns), '\n')

# 显示数据集的前五行
display(data_dummies.head())

# 定义数据集的特征值
features = data_dummies.loc[:, '年龄':'职业_ Transport-moving']  # 先行后列
# 将特征数值赋值为X
X = features.values
# 将收入大于50k作为预测目标
y = data_dummies['收入_ >50K'].values
# 打印数据形态
print('特征形态：{} 标签形态{}'.format(X.shape, y.shape))

# 拆分
X_train, X_test, y_train, y_test = train_test_split(X, y)
go_dating_tree = tree.DecisionTreeClassifier(max_depth=5)
go_dating_tree.fit(X_train, y_train)
print('{:.2f}'.format(go_dating_tree.score(X_test, y_test)))

# 将Mr_Z的数据输入模型
Mr_Z = [
    [37, 40, 0, 0, 0, 0, 0, 0, 1, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
     1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0]]
# 用模型做出预测
dating_dec = go_dating_tree.predict(Mr_Z)
if dating_dec == 1:
    print('yes')
else:
    print('no')
