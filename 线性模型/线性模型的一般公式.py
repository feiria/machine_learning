import numpy as np
import matplotlib.pyplot as plt
# 令x为-5到5之间，元素数为100的等差数列
x = np.linspace(-5, 5, 100)
# 输入直线方程
y = 0.5*x + 3
plt.plot(x, y, c='orange')
# 图题设为‘Straight Line’
plt.title('Straight Line')
plt.show()