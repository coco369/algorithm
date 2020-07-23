"""
    参考地址：https://www.jianshu.com/p/9879bdce0a60
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

# 常用激活函数
# relu函数，公式f(x) = max(x, 0)
y_relu = tf.nn.relu(x)

y_sigmoid = tf.nn.sigmoid(x)
y_tanh = tf.nn.tanh(x)
# softplu函数，是relu函数的平滑版本，公式f(x) = log(1 + e^x)
y_softplus = tf.nn.softplus(x)
# y_softmax = tf.nn.softmax(x)  softmax is a special kind of activation function, it is about probability

# plt to visualize these activation function
plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()
