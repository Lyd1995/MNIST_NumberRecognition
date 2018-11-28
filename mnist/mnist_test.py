# coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
import matplotlib.pyplot as plt

TEST_INTERVAL_SECS = 3


def test(mnist):
    with tf.Graph().as_default() as g:  # 使用数据流图
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None)   # 开始前向传播，得到输出值y

        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)  # 滑动平均
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)  # 保存

        # tf.equal():比较两个矩阵的对应元素是否相等，相等返回true
        # tf.argmax(x, axis):取最大值,0:求列的最大值，1：求行的最大值
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))    # 比较：标签与预测值
        # tf.cast(x, dtype)：将参数x转换为指定的类型dtype
        # tf.reduce_mean(x, axis):取张量指定维度的平均值。
        # 不指定第二参数时，则在所有元素中取平均值，0：在第一维元素中取平均值，即每一列求平均。1:在第二维元素上取平均，即每一行取平均
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 定义准确度,观察所得神经网络的准确度
        accuracy_y = []
        accuracy_x = []
        global_step_old = 0

        while True:
            with tf.Session() as sess:  # 运行计算图
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # print(global_step, type(global_step))
                    if global_step_old == int(global_step):
                        break
                    global_step_old = int(global_step)
                    # print(global_step_old, type(global_step_old))
                    # 每次输入的数据来自于测试集
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                    accuracy_x.append(int(global_step))
                    accuracy_y.append(accuracy_score)
                else:
                    print('No checkpoint file found')
                    plt.plot(accuracy_x, accuracy_y)
                    plt.show()
                    return
            time.sleep(TEST_INTERVAL_SECS)

        plt.figure(1)
        plt.plot(accuracy_x, accuracy_y)
        plt.xlabel("global_step")
        plt.ylabel("accuracy")
        plt.show()
        return

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)


if __name__ == '__main__':
    main()


"""
    print("train data size:", mnist.train.num_examples)
    print("validation data size:", mnist.validation.num_examples)
    print("test data size:", mnist.test.num_examples)
    print(mnist.train.labels[0])
"""
