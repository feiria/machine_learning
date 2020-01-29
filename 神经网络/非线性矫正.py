import numpy as np
import matplotlib.pyplot as plt

line = np.linspace(-5, 5, 200)

plt.plot(line, np.tanh(line), label='tanh')
plt.plot(line, np.maximum(line, 0), label='relu')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('relu(x) and tanh(x)')
plt.show()