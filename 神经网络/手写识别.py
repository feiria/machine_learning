# 导入数据集获取工具
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from PIL import Image
import numpy as np

# 加载MNIST手写数字数据集
mnist = fetch_mldata('MNIST original', data_home='./')
print(mnist)
print('样本数量{} 样本特征数{}'.format(mnist.data.shape[0], mnist.data.shape[1]))

X = mnist.data / 255.
y = mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000, test_size=1000, random_state=62)

mlp_hw = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[100, 100], activation='relu', alpha=1e-5, random_state=62)
mlp_hw.fit(X_train, y_train)
print('score: {:.2f}'.format(mlp_hw.score(X_test, y_test) * 100))

image = Image.open('4.png').convert('F')
image = image.resize((28, 28))
arr = []
for i in range(28):
    for j in range(28):
        pixel = 1.0 - float(image.getpixel((j, i))) / 255.
        arr.append(pixel)

# 由于只有一个样本， 所以需要进行reshape
arr1 = np.array(arr).reshape(1, -1)
print('{:.0f}'.format(mlp_hw.predict(arr1)[0]))
