import tensorflow as tf
import matplotlib.pyplot as plt


class Model():
    def __init__(self):
        self.W = tf.Variable(10.0)
        self.b = tf.Variable(-5.0)

    # 将Model对象当作函数使用的时候，将调用__call__方法
    def __call__(self, inputs):
        # 相当于函数 y=常量*x + 常数项
        return self.W * inputs + self.b


def compute_loss(y_true, y_pred):
    """
        根据实际值和预测值求出均方误差，误差越小，则预测越准确
        y_true: 真实值
        y_pred：预估值
        tf.reduce_mean(): 用于计算列表中数据求和之后在求平均值
        tf.square(): 用于计算参数的平方
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))


model = Model()

# 定义权重Weight和偏差bias
TRUE_W = 5.0
TRUE_b = 2.0

# 获取训练数据，将训练数据与噪声进行合成
NUM_EXAMPLES = 2000
inputs = tf.random.normal(shape=[NUM_EXAMPLES])
noise = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise
# outputs = 2*inputs + TRUE_b


# 开始训练数据
# 蓝色为训练数据，红色为训练结果

def plot(epoch):
    """
        plt.scatter(x, y): 描绘散列图，x为横坐标，y为纵坐标
    """
    plt.scatter(inputs, outputs, c='b')
    plt.scatter(inputs, model(inputs), c='r')

    plt.title('epoch %s, loss=%s' % (epoch, str(compute_loss(outputs, model(inputs)).numpy())))
    plt.legend()
    plt.draw()
    plt.ion()
    plt.pause(2)
    plt.close()


learning_rate = 0.1
for epoch in range(30):
    with tf.GradientTape(persistent=True) as t:
        t.watch([model.W, model.b])
        # GradientTape：梯度带，在with中的所有过程将会被记录
        loss = compute_loss(outputs, model(inputs))
        print('损失值：', loss)

    # gradient(ys, xs): ys类似是张量或者张量列表，类似于目标函数，需要被微分的函数
    #                   xs类似是张量或者张量列表，需要求微分的对象。
    # 微分结果为 dys/dxs
    dW, db = t.gradient(loss, [model.W, model.b])
    print('权重dW', dW)
    print('常量db', db)
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

    plot(epoch + 1)
