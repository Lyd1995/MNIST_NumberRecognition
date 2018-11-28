import tensorflow as tf
from PIL import Image
import mnist_forward
import mnist_backward
import numpy as np
import matplotlib.pyplot as plt



# 对输入图像做处理
def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    threshold = 50
    # 进行二值化处理
    for i in range(28):
        for j in range(28):
            if (im_arr[i][j] < threshold):  # 设置一个阈值，小于这个阈值的像素默认为黑色
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255          # 大于的则设为白色

    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)  # 做归一化处理，将0到255之间的数变为0到1之间的浮点数
    return img_ready, img


# 加载训练好的模型
def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print('No checkpoint file found')
                return -1


def application():
    testNum = eval(input("input the number of test pictures:"))
    for i in range(testNum):
        testPic = "testpic/" + str(i) + ".jpg"
        testPicArr, img = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        plt.figure(i)
        plt.imshow(img)
        plt.title(str(preValue))
        plt.show()


if __name__ == "__main__":
    application()
