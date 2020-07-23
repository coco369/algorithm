import tensorflow as tf

"""
    tensorflow的基本操作
"""

a = tf.constant(3.0)
print(a.numpy())

# tf.GradientTape 自动微分的api
# persistent参数：持久性，如果不设置则gradient只能调用一次，就通过垃圾回收机制释放内存了
with tf.GradientTape(persistent=True) as tt:
    tt.watch(a)

    with tf.GradientTape(persistent=True) as t:
        t.watch(a)

        a1 = tf.Variable(3.0)
        print(a1.numpy())

        y = a * a
        z = y * y
        print(y.numpy())  # 9.0
        print(z.numpy())  # 81.0

        # 相当于z=a^4，一阶求导
        # 一阶求导为 1 * 4 * a^3 = 4 * 27.0 = 108.0
        dz_da = t.gradient(z, a)
        print(dz_da.numpy())  # 108.0

        # 相当于 y=a^2, 一阶求导
        # dy_da = 2 * 1 * a
        dy_da = t.gradient(y, a)
        print(dy_da.numpy())  # 6.0

    # 相当于二阶求导
    # 二阶导数是一阶导数的导数
    dy2_da2 = t.gradient(dy_da, a)
    print(dy2_da2.numpy())

del t


# 二元梯度
a3 = tf.constant(3.0)
b3 = tf.constant(2.0)

with tf.GradientTape(persistent=True) as t:
    t.watch([a3, b3])
    z = 2 * a3 * a3 * b3

g1, g2 = t.gradient(z, [a3, b3])
# 微分函数为：y=2x^2 * z
print(g1, g2)



