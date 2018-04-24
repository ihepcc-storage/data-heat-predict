# -*-coding:UTF-8-*-

import tensorflow as tf

#定义神经网络相关参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weigth_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))

    return weights


def cond()
