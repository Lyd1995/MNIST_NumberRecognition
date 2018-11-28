import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


# 获取参数w
def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


# 获取偏置参数b
def get_bias(shape):  
    b = tf.Variable(tf.zeros(shape))  
    return b


# 定义前向传播过程
def forward(x, regularizer):
    # 第一层神经网络
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)  # 赋予参数w随机初值，矩阵大小：[INPUT_NODE, LAYER1_NODE]
    b1 = get_bias([LAYER1_NODE])                             # 赋予偏置的初值为0
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)                   # 激活函数采用Relu函数

    # 第二层神经网络
    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)   # 赋予参数w随机初值
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    return y
