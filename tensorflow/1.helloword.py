import tensorflow as tf

"""
    定义常数：constant
"""

helloworld = tf.constant('hello word', tf.string)

print('tensor:', helloworld)
print('value', helloworld.numpy())
