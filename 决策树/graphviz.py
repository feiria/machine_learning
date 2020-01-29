# 导入graphviz工具
import graphviz
# 导入决策树中输出graphviz的接口
from sklearn.tree import export_graphviz

# 打开一个dot文件
with open("wine.dot") as f:
    dot_graph = f.read()
    # 显示dot文件中的图形
    graphviz.Source(dot_graph)