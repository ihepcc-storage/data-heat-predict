# -*-coding:UTF-8-*-
import sys, os
import tensorflow as tf
import numpy as np
import BuildDataSet01


# Hyper Parameters
learning_rate = 0.001   #学习率
n_steps = 100            #LSTM 展开步数
n_inputs = 24           #输入节点数
n_hiddens = 256         #隐层节点数
n_layers = 4          #LSTM layer层数
n_classes = 4         #输出节点数（分类数目）

#LossWeight = [1000, 1000, 1000, 1]

#LossSess = tf.Session()
# data
"""
DataSets = BuildDataSet.read_data_sets(train_time_end=9999999999.99-1520611200.00, train_time_begin=9999999999.99-1521043200.00,
                   test_time_end=9999999999.99-1521129600.00, test_time_begin=9999999999.99-1521475200.00)
train = DataSets[0]
validation = DataSets[1]
test = DataSets[2]
test_x = test.features
test_y = test.labels
"""

# tensor placeholder
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, n_steps * n_inputs], name='x_input')
    y = tf.placeholder(tf.float32, [None, n_classes], name='y_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob_input')           # 保持多少不被dropout
    batch_size = tf.placeholder(tf.int32, [], name='batch_size_input')        # 批大小


# weights and bias
with tf.name_scope('weights'):
    weights = tf.Variable(tf.truncated_normal([n_hiddens, n_classes],stddev=0.1), dtype=tf.float32, name='W')
    tf.summary.histogram('output_layer_weights', weights)
with tf.name_scope('bias'):
    biases = tf.Variable(tf.random_normal([n_classes]), name='b')
    tf.summary.histogram('output_layer_bias', biases)


# RNN structure
def RNN_LSTM(x, weights, biases):
    #  RNN 输入 reshape
    x = tf.reshape(x, [-1, n_steps, n_inputs])
    #  定义 LSTM cell
    #  cell 中的 dropout
    def attn_cell():
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hiddens)
        with tf.name_scope('lstm_dropout'):
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

    # 实现多层 LSTM
    enc_cells = []
    for i in range(0, n_layers):
        enc_cells.append(attn_cell())
    with tf.name_scope('lstm_cells_layers'):
        mlstm_cell = tf.contrib.rnn.MultiRNNCell(enc_cells, state_is_tuple=True)

    # 全零初始化 state
    _init_state = mlstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    # dynamic_rnn  运行网络
    outputs, states = tf.nn.dynamic_rnn(mlstm_cell, x, initial_state=_init_state, dtype=tf.float32, time_major=False)
    # 输出
    #print tf.nn.softmax(tf.matmul(outputs[:,-1,:], weights)+biases)
    return tf.nn.softmax(tf.matmul(outputs[:,-1,:], weights)+biases)

with tf.name_scope('output_layer'):
    pred = RNN_LSTM(x,weights,biases)
    tf.summary.histogram('outputs', pred)
# cost
with tf.name_scope('loss'), tf.Session() as LossSess:

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y
                                            ))
    #cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred),reduction_indices=[1]))
    """
    predArray = LossSess.run(pred)
    labelArray = LossSess.run(y)
    _predArray = np.empty(shape=[0, 4])
    _lableArray = np.empty(shape=[0, 4])
    for i in range(0,labelArray.shape[0]):
        if labelArray[0] == np.array([1,0,0,0],dtype=float):
            for j in range(0, LossWeight[0]):
                np.concatenate((_predArray, predArray[i]),axis=0)
                np.concatenate((_lableArray, labelArray[i]),axis=0)
        if labelArray[0] == np.array([0,1,0,0],dtype=float):
            for j in range(0, LossWeight[1]):
                np.concatenate((_predArray, predArray[i]),axis=0)
                np.concatenate((_lableArray, labelArray[i]),axis=0)
        if labelArray[0] == np.array([0,0,1,0],dtype=float):
            for j in range(0, LossWeight[2]):
                np.concatenate((_predArray, predArray[i]),axis=0)
                np.concatenate((_lableArray, labelArray[i]),axis=0)
        if labelArray[0] == np.array([0,0,0,1],dtype=float):
            for j in range(0, LossWeight[3]):
                np.concatenate((_predArray, predArray[i]),axis=0)
                np.concatenate((_lableArray, labelArray[i]),axis=0)
    _pred = tf.convert_to_tensor(_predArray,dtype=float)
    _label = tf.convert_to_tensor(_lableArray,dtype=float)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred, labels=_label))
    """
    tf.summary.scalar('loss', cost)
# optimizer
with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
with tf.name_scope('accuracy'):
    #accuracy = tf.metrics.accuracy(labels=tf.arg_max(y, 1), predictions=tf.arg_max(pred,1))[1]
    #tf.summary.scalar('accuracy', accuracy)

    correct_prediction = tf.equal(tf.arg_max(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



#init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())



with tf.Session() as sess:
    merged = tf.summary.merge_all()


    features = np.empty(shape=[0, 2400])
    labels = np.empty(shape=[0, 4])
    #
    dataset = BuildDataSet01.DataSet(features=features, labels=labels, train_time_begin=9999999999.99 - 1521993600.00,
                      train_time_end=9999999999.99 - 1521475200.00, PredictTimeOffset=86400.00,
                      PredictTimeWindow=86400.00*7, n_steps=100,
                      capacity=10000, limit=1000)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    train_writer = tf.summary.FileWriter("./logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("./logs/test", sess.graph)

    # training
    i = 0
    while True:
        i += 1
        _batch_size = 100
        batch_x, batch_y = dataset.next_batch_series(batch_size=_batch_size)
        #print 'a new batch'
        if batch_x==np.array([-1]):
            break

        if (i+1) % 1==0:
            #test_x, test_y = dataset.next_batch_series(batch_size=16)
            test_x, test_y = batch_x, batch_y
            train_accuracy = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.0,
                                                                     batch_size: test_x.shape[0]})
            print "Testing Accuracy:", train_accuracy
            #print 'iter:',i,'loss:',cost
            #train_result = sess.run(merged, feed_dict={x:batch_x, y:batch_y, keep_prob:1.0, batch_size:_batch_size})
            #test_result = sess.run(merged, feed_dict={x:test_x, y:test_y, keep_prob:1.0, batch_size:test_x.shape[0]})
            #train_writer.add_summary(train_result, i+1)
            #test_writer.add_summary(test_result, i+1)

        _, loss_ = sess.run([train_op, cost],
                            feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5, batch_size: _batch_size})
        print 'iter:', i, 'loss:', loss_

        rs = sess.run(merged)
        train_writer.add_summary(rs, i)

    print "Optimization Finished!"

    # prediction
    # print "Testing Accuracy:", sess.run(accuracy, feed_dict={x:test_x, y:test_y, keep_prob:1.0, batch_size:test_x.shape[0]})




