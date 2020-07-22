import tensorflow as tf

"""
    定义变量
"""

a = tf.Variable(1)
print('value:', a.numpy())

b = a + 1
print('value:', b.numpy())

c = tf.Variable(4)
c.assign_add(2)
print('value:', c.numpy())

c.assign_sub(1)
print('value:', c.numpy())
