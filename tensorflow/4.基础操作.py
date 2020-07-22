import tensorflow as tf


def f(x, y):
    output = 1.0
    for i in range(y):
        if 1 < i < 5:
            output = tf.multiply(output, x)
    # output: x^3
    return output


def grade(x, y):
    with tf.GradientTape(persistent=True) as t:
        t.watch(x)
        out = f(x, y)
    # 相当于y=x^3 对x进行一阶求导，值为3*x^2 = 12.0
    return t.gradient(out, x)


x = tf.convert_to_tensor(2.0)
print(x)
print(x.numpy())

print(grade(x, 6).numpy())
print(grade(x, 5).numpy())
