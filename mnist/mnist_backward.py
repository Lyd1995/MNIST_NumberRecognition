import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1      # 学习率初值
LEARNING_RATE_DECAY = 0.99    # 学习率的衰减率
REGULARIZER = 0.0001          # 正则化参数
STEPS = 50002                 # 迭代训练轮数
MOVING_AVERAGE_DECAY = 0.99   # 滑动平均衰减率
MODEL_SAVE_PATH = "./model/"  # 保存路径（主要是相关数据流图及参数）
MODEL_NAME = "mnist_model"    # 保存名称


def backward(mnist):

    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])   # 输出的预测结果
    y = mnist_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)                        # 当前训练的轮次

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))          # 定义损失函数

    learning_rate = tf.train.exponential_decay(                 # 采用指数衰减学习率
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, 
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)  # 使用梯度下降算法

    # 滑动平均值 = 衰减率*滑动平均值 + （1 - 衰减率）*参数， MOVING_AVERAGE_DECAY：平均衰减率
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    ema_op = ema.apply(tf.trainable_variables())         # tf.trainable_variables()：把所有待训练参数汇总为列表
    with tf.control_dependencies([train_step, ema_op]):  # 这个函数实现滑动平均和训练过程同步运行
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()  # 保存神经网络的模型

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        loss_value_y = []
        loss_value_x = []

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)  # 每次训练,从训练集中输入BATCH_SIZE个训练样本
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                loss_value_x.append(step)
                loss_value_y.append(loss_value)
                # 将神经网络中的所有参数保存到指定路径
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        plt.figure(2)
        plt.plot(loss_value_x, loss_value_y)
        plt.xlabel("steps")
        plt.ylabel("loss")
        plt.show()

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)  # 获取训练集
    backward(mnist)


if __name__ == '__main__':
    main()


