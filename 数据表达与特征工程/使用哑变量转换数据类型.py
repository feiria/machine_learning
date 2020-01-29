import pandas as pd
from IPython.core.display import display

fruits = pd.DataFrame({
    '数值特征': [5, 6, 7, 8, 9],
    '类型特征': ['西瓜', '香蕉', '橘子', '苹果', '葡萄']
})
display(fruits)

fruits_dum = pd.get_dummies(fruits)
display(fruits_dum)

fruits['数值特征'] = fruits['数值特征'].astype(str)
display(pd.get_dummies(fruits, columns=['数值特征']))