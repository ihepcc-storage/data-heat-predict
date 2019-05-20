#!/bin/env python
# -*-coding:UTF-8-*-

import sys, os
import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np
import random
import pandas as pd
from pandas import Series, DataFrame
from scipy import sparse
from sklearn.utils import resample
from sklearn.utils import shuffle


class DataSet(object):

    def __init__(self,features,labels,one_hot=False,zoom=False,MaxValue=0,dtype=dtypes.float32):
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, excepted uint8 or float32' % dtype)
        if dtype == dtypes.float32:
            features = features.astype(np.float32)
        if zoom:
            features = features.multiply(features, 1.0/MaxValue)
        self._features = features
        self._num_examples = features.shape[0]
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        pass

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def random_sample_batch(self, batch_size):
        perm0 = np.arange(self._num_examples)
        slice = random.sample(perm0, batch_size)
        return self.features[slice], self.labels[slice]

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start==0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._features = self.features[perm0]
            self._labels = self.labels[perm0]
        #Go to the next epoch
        if start + batch_size > self._num_examples:
            #Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            features_rest_part = self._features[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._features = self.features[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            features_new_part = self._features[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((features_rest_part,features_new_part), axis=0), \
                   np.concatenate((labels_rest_part,labels_new_part),axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            #print self._index_in_epoch, batch_size
            return self._features[start:end], self.labels[start:end]
        pass

"""
# read DataSet
def read_zip_csv(zip_csv_list=None):
    DataframeList = []
    for zip_csv in zip_csv_list:
        try:
            z_csv = zipfile.ZipFile(zip_csv)
        except:
            print 'Open', zip_csv, 'error!'
            continue
        for csvFile in set(z_csv.namelist()):
            data = pd.read_csv(z_csv.open(csvFile), dtype='float', header=None)

            if data.isnull().values.any():
                print 'NaN value exist in dataframe data:', np.nonzero(data.isnull().any(1))[0].tolist()
                #print data[data.isnull().any(1)]
                # a = np.nonzero(data.isnull().any(1))[0].tolist()
                data = data.dropna()
                print np.nonzero(data.isnull().any(1))[0].tolist()
                #data = data.drop(a, axis=0)
            DataframeList.append(data)
            print data.values.shape
        z_csv.close()
    return pd.concat(DataframeList, axis=0)
    pass
"""

# read Dataset
def read_sparse_matrix(sparse_matrix_list=None, bootstrap=False, SampleCount=-1):
    if not bootstrap:
        SampleCount = -1
    DataframeList = []
    for _sparse_matrix_list in sparse_matrix_list:
        feature_matrix = None
        _matrix = None
        ct=0
        for sparse_matrix in _sparse_matrix_list:
            print 'Read ', sparse_matrix
            ct += 1
            try:
                matrix = sparse.load_npz(sparse_matrix)
            except:
                print 'Open', sparse_matrix, 'error!'
                continue
            try:
                _matrix = sparse.vstack((_matrix,matrix))
            except:
                _matrix = matrix
            if ct%100==0:
                try:
                    feature_matrix = sparse.vstack((feature_matrix, _matrix))
                    _matrix = None
                except:
                    feature_matrix = _matrix
        try:
            if _matrix!=None:
                feature_matrix = sparse.vstack((feature_matrix, _matrix))
                _matrix = None
        except:
            feature_matrix = _matrix
        if feature_matrix==None:
            continue
        feature_matrix = resample(feature_matrix,replace=True,n_samples=SampleCount,random_state=1)
        df = DataFrame(feature_matrix.toarray())
        DataframeList.append(df)
    return DataframeList
    return pd.concat(DataframeList, axis=0)
    pass

"""
def build_zip_csv_list(zip_csv_class01_prefix,zip_csv_class01_begin_index, zip_csv_class01_end_index,
                       zip_csv_class02_prefix, zip_csv_class02_begin_index, zip_csv_class02_end_index,
                       zip_csv_class03_prefix, zip_csv_class03_begin_index, zip_csv_class03_end_index):
    zip_csv_list = []
    _list1, _list2, _list3 = [], [], []
    for i in range(zip_csv_class01_begin_index, zip_csv_class01_end_index+1):
        _list1.append(zip_csv_class01_prefix+str(i))
    zip_csv_list.append(_list1)
    for i in range(zip_csv_class02_begin_index, zip_csv_class02_end_index+1):
        _list2.append(zip_csv_class02_prefix+str(i))
    zip_csv_list.append(_list2)
    for i in range(zip_csv_class03_begin_index, zip_csv_class03_end_index+1):
        _list3.append(zip_csv_class03_prefix+str(i))
    zip_csv_list.append(_list3)
    return zip_csv_list
"""
def build_sparse_matrix_list(sparse_matrix_class01_format,sparse_matrix_class01_begin_index,sparse_matrix_class01_end_index,
                             sparse_matrix_class02_format, sparse_matrix_class02_begin_index,
                             sparse_matrix_class02_end_index,sparse_matrix_class03_format,
                             sparse_matrix_class03_begin_index,sparse_matrix_class03_end_index,
                             sparse_matrix_class04_format, sparse_matrix_class04_begin_index,
                             sparse_matrix_class04_end_index):
    sparse_matrix_list = []
    _list1, _list2, _list3, _list4 = [], [], [], []
    for i in range(sparse_matrix_class01_begin_index, sparse_matrix_class01_end_index+1):
        _list1.append(sparse_matrix_class01_format.replace('*', str(i), 1))
    sparse_matrix_list.append(_list1)
    for i in range(sparse_matrix_class02_begin_index, sparse_matrix_class02_end_index+1):
        _list2.append(sparse_matrix_class02_format.replace('*', str(i), 1))
    sparse_matrix_list.append(_list2)
    for i in range(sparse_matrix_class03_begin_index, sparse_matrix_class03_end_index+1):
        _list3.append(sparse_matrix_class03_format.replace('*', str(i), 1))
    sparse_matrix_list.append(_list3)
    for i in range(sparse_matrix_class04_begin_index, sparse_matrix_class04_end_index+1):
        _list4.append(sparse_matrix_class04_format.replace('*', str(i), 1))
    sparse_matrix_list.append(_list4)
    return sparse_matrix_list

"""
zip_csv_list = build_zip_csv_list('/scratchfs/cc/chengzj/csv/LSTM_TRAIN_DATASET_class01.csv.zip.',0,1,
                                  '/scratchfs/cc/chengzj/csv/LSTM_TRAIN_DATASET_class02.csv.zip.',0,0,
                                  '/scratchfs/cc/chengzj/csv/LSTM_TRAIN_DATASET_class03.csv.zip.',0,170)
print zip_csv_list

TrainDataList = []
Sum_Len = 0
Max_Index = -1
Max_Len = 0
for i in zip_csv_list:
    _dataframe = read_zip_csv(zip_csv_list=i)
    print 'TrainFeatureSub.shape:', _dataframe.values.shape
    Sum_Len += _dataframe.values.shape[0]
    if _dataframe.values.shape[0]>Max_Len:
        Max_Len = _dataframe.values.shape[0]
        Max_Index = zip_csv_list.index(i)
    TrainDataList.append(_dataframe)
#TrainData = read_zip_csv(zip_csv_list=zip_csv_list)
TrainData = pd.concat(TrainDataList, axis=0)
TrainFeature = TrainData.values[:,:-3]
TrainLabel = TrainData.values[:,-3:]
"""

sparse_matrix_list = build_sparse_matrix_list('./csv01/LSTM_TRAIN_DATASET_class01.*.npz',1,1794,
                                              './csv01/LSTM_TRAIN_DATASET_class02.*.npz',1,1,
                                              './csv01/LSTM_TRAIN_DATASET_class03.*.npz',1,0,
                                              './csv01/LSTM_TRAIN_DATASET_class04.*.npz',1,2)
TrainDataList = read_sparse_matrix(sparse_matrix_list,bootstrap=True,SampleCount=100000)
TrainData = pd.concat(TrainDataList, axis=0)

TrainFeature = TrainData.values[:,0:4320]
TrainLabel = TrainData.values[:,4321:-2]

print 'After Bootstrapping TrainFeature.shape:', TrainFeature.shape, 'TrainLable.shape:', TrainLabel.shape
TrainDataset = DataSet(features=TrainFeature, labels=TrainLabel)


"""
zip_csv_list = build_zip_csv_list('/scratchfs/cc/chengzj/csv01/LSTM_TRAIN_DATASET_class01.csv.zip.',0,6,
                                  '/scratchfs/cc/chengzj/csv01/LSTM_TRAIN_DATASET_class02.csv.zip.',0,1,
                                  '/scratchfs/cc/chengzj/csv01/LSTM_TRAIN_DATASET_class03.csv.zip.',0,113)
print zip_csv_list
TestDataList = []
Sum_Len = 0
Max_Index = -1
for i in zip_csv_list:
    _dataframe = read_zip_csv(zip_csv_list=i)
    print 'TestFeatureSub.shape:', _dataframe.values.shape
    Sum_Len += _dataframe.values.shape[0]
    #if _dataframe.values.shape[0]>Max_Len:
        #Max_Len = _dataframe.values.shape[0]
        #Max_Index = zip_csv_list.index(i)
    TestDataList.append(_dataframe)


TestFeatureList, TestLabelList = [], []
for dataframe in TestDataList:
    TestFeatureList.append(dataframe.values[:,:-3])
    TestLabelList.append(dataframe.values[:,-3:])
TestDatasetList = []
for i in range(0, len(TestFeatureList)):
    print 'TestDataSet sub', i, 'length', TestFeatureList[i].shape[0], TestLabelList[i].shape[0]
    TestDatasetList.append(DataSet(features=TestFeatureList[i], labels=TestLabelList[i]))


TestDataListCopy = TestDataList
for i in range(0,len(TestDataListCopy)):
    #if i == Max_Index:
        #print i, 'no need'
        #continue
    TestDataListCopy[i] = resample(TestDataListCopy[i], replace=True, n_samples=Sum_Len/3, random_state=234)


TestData = pd.concat(TestDataListCopy, axis=0)
TestFeature = TestData.values[:,:-3]
TestLabel = TestData.values[:,-3:]

print 'After Bootstrapping TestFeature.shape:', TestFeature.shape, 'TestLable.shape:', TestLabel.shape
TestDataset = DataSet(features=TestFeature, labels=TestLabel)
"""




# Hyper Parameters
# learning_rate = 0.001   #学习率
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.01       #250
learning_rate = starter_learning_rate
train_batch_size = 256
test_batch_size = 1000

n_steps = 24*30            #LSTM 展开步数
n_inputs = 6           #输入节点数
n_hiddens = 256         #隐层节点数
n_layers = 3          #LSTM layer层数
n_classes = 4         #输出节点数（分类数目）

"""
with tf.name_scope('learning_rate'):
    learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,decay_steps=TrainFeature.shape[0]/train_batch_size,
                                                decay_rate=0.8,staircase=True)
    learning_rate_scalar = tf.summary.scalar('learning_rate',learning_rate)
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
    loss_scalar = tf.summary.scalar('loss', cost)
# optimizer
with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

# correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
with tf.name_scope('accuracy'):
    accuracy = tf.metrics.accuracy(labels=tf.arg_max(y, 1), predictions=tf.arg_max(pred,1))[1]
    accuracy_scalar = tf.summary.scalar('accuracy', accuracy)
    #correct_prediction = tf.equal(tf.arg_max(pred,1),tf.argmax(y,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
with tf.name_scope('accuracy_class0'):
    accuracy_class0 = tf.metrics.accuracy(labels=tf.arg_max(y, 1), predictions=tf.arg_max(pred, 1))[1]
    accuracy_class0_scalar = tf.summary.scalar('accuracy_class0', accuracy_class0)
with tf.name_scope('accuracy_class1'):
    accuracy_class1 = tf.metrics.accuracy(labels=tf.arg_max(y, 1), predictions=tf.arg_max(pred, 1))[1]
    accuracy_class1_scalar = tf.summary.scalar('accuracy_class1', accuracy_class1)
with tf.name_scope('accuracy_class2'):
    accuracy_class2 = tf.metrics.accuracy(labels=tf.arg_max(y, 1), predictions=tf.arg_max(pred, 1))[1]
    accuracy_class2_scalar = tf.summary.scalar('accuracy_class2', accuracy_class2)
with tf.name_scope('accuracy_class3'):
    accuracy_class3 = tf.metrics.accuracy(labels=tf.arg_max(y, 1), predictions=tf.arg_max(pred, 1))[1]
    accuracy_class3_scalar = tf.summary.scalar('accuracy_class3', accuracy_class3)


saver = tf.train.Saver()

with tf.Session() as sess:
    #merged = tf.summary.merge_all()
    #features = np.empty(shape=[0, 1176])
    #labels = np.empty(shape=[0, 3])

    train_writer = tf.summary.FileWriter("./logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("./logs/test", sess.graph)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # restore LSTM patameters
    """
    try:
        saver.restore(sess, 'Model/LSTM')
    except Exception as e:
        print e
    """

    # training
    i = 0
    last_train_accuracy = 0
    batch_x, batch_y = None, None
    batch_x, batch_y = TrainDataset.next_batch(batch_size=train_batch_size, shuffle=True)
    while True:
        i += 1

        _, loss_ = sess.run([train_op, loss_scalar],
                            feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5, batch_size: batch_x.shape[0]})
        train_writer.add_summary(loss_, i)

        batch_x, batch_y = TrainDataset.next_batch(batch_size=train_batch_size, shuffle=True)
        print 'iter:', i, '_epochs_completed:', TrainDataset._epochs_completed

        if (i+1) % 10==0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0,
                                                             batch_size: batch_x.shape[0]})
            print "Train Accuracy(use next batch):", train_accuracy


        if (i+1) % 10==0:
            if train_accuracy<last_train_accuracy:
                if learning_rate*0.8>0.001:
                    learning_rate = learning_rate*0.8
                else:
                    learning_rate = starter_learning_rate
            last_train_accuracy = train_accuracy
            print 'learning_rate:', learning_rate


        if (i+1) % 1000==0:
            batch_x, batch_y = TrainDataset.random_sample_batch(batch_size=test_batch_size)
            train_accuracy_metall = sess.run(accuracy_scalar, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0,
                                                                        batch_size: batch_x.shape[0]})
            #print "Train Accuracy(use train dataset sample):", train_accuracy
            train_writer.add_summary(train_accuracy_metall, i)

        '''
        if (i+1) % 1000==0:
            """
            learning_rate_metall = sess.run(learning_rate_scalar)
            #print 'learning_rate:', learning_rate_metall
            train_writer.add_summary(learning_rate_metall, i)
            """

            test_batch_x, test_batch_y = TestDataset.next_batch(batch_size=test_batch_size, shuffle=True)
            test_accuracy_metall = sess.run(accuracy_scalar, feed_dict={x: test_batch_x, y: test_batch_y, keep_prob: 1.0,
                                                                     batch_size: test_batch_x.shape[0]})
            print 'accuracy:', sess.run(accuracy, feed_dict={x: test_batch_x, y: test_batch_y, keep_prob: 1.0,
                                                                     batch_size: test_batch_x.shape[0]})
            #print test_batch_x, test_batch_y
            #print "Test Accuracy:", test_accuracy_metall
            test_writer.add_summary(test_accuracy_metall, i)

            for item in range(0, len(TestDatasetList)):
                _test_batch_x, _test_batch_y = TestDatasetList[item].next_batch(batch_size=test_batch_size, shuffle=True)
                _test_accuracy = 0
                if item==0:
                    _test0_accuracy_metall = sess.run(accuracy_class0_scalar, feed_dict={x: _test_batch_x, y: _test_batch_y, keep_prob: 1.0,
                                                                         batch_size: _test_batch_x.shape[0]})
                    print 'class 0 accuracy:', sess.run(accuracy_class0, feed_dict={x:_test_batch_x, y:_test_batch_y, keep_prob:1.0, batch_size:_test_batch_x.shape[0]})
                    test_writer.add_summary(_test0_accuracy_metall, i)
                    #print "Testing class", item, "Accuracy:", sess.run(accuracy_class0)
                if item==1:
                    _test1_accuracy_metall = sess.run(accuracy_class1_scalar, feed_dict={x: _test_batch_x, y: _test_batch_y, keep_prob: 1.0,
                                                                         batch_size: _test_batch_x.shape[0]})
                    print 'class 1 accuracy:', sess.run(accuracy_class1, feed_dict={x:_test_batch_x, y:_test_batch_y, keep_prob:1.0, batch_size:_test_batch_x.shape[0]})
                    test_writer.add_summary(_test1_accuracy_metall, i)
                    #print "Testing class", item, "Accuracy:", sess.run(accuracy_class1)
                if item==2:
                    _test2_accuracy_metall = sess.run(accuracy_class2_scalar, feed_dict={x: _test_batch_x, y: _test_batch_y, keep_prob: 1.0,
                                                                         batch_size: _test_batch_x.shape[0]})
                    print 'class 2 accuracy:', sess.run(accuracy_class2, feed_dict={x:_test_batch_x, y:_test_batch_y, keep_prob:1.0, batch_size:_test_batch_x.shape[0]})
                    test_writer.add_summary(_test2_accuracy_metall, i)
                    #print "Testing class", item, "Accuracy:", sess.run(accuracy_class2)
        '''


        if (i+1) % 1000 ==0:
            saver.save(sess, 'Model/LSTM')


    print "Optimization Finished!"

    # prediction
    # print "Testing Accuracy:", sess.run(accuracy, feed_dict={x:test_x, y:test_y, keep_prob:1.0, batch_size:test_x.shape[0]})


