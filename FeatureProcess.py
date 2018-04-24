# -*-coding:UTF-8-*-
import sys, os
import logging
import config
from collections import OrderedDict
import collections
import Math.stats
import numpy
import scipy
import scipy.stats as scista
#import pywt
import heapq
import tempfile
import MysqlCon
#import dtypes
#import matplotlib.pyplot as plt

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed



total_path = 0
total_path_dict = {}
path_td_feature_dict = OrderedDict()
feature_shape_list = []
feature_shape_array = None
label_shape_array = None

train_feature_shape_array = None
train_label_shape_array = None

test_feature_shape_array = None
test_label_shape_array = None


def listZeroLen(x):
    len = 0
    _len = 0
    try:
        for t in x:
           if t==0:
               len+=1
           else:
               if len>_len:
                   _len = len
               len = 0
        if len>_len:
            _len = len
        return _len
    except:
        return 0

def FindPeak(x):
    return heapq.nlargest(3,enumerate(x), key=lambda x: x[1])

'''
_tmp_list = []
def featurelist_to_list(feature_list,beginIndex=0):
    global feature_shape_list
    for t in feature_list[beginIndex]:
        _tmp_list.append(t)
        if (len(feature_list)-1)>beginIndex:
            featurelist_to_list(feature_list,beginIndex+1)
        else:
            del _tmp_list[-1]
            for k in feature_list[beginIndex]:
                _tmp_list.append(k)
                feature_shape_list.extend(_tmp_list)
                del _tmp_list[-1]
        del _tmp_list[-1]

    pass
'''

#def ProcessByHour(HoursBefore=24):
def extract_features(TimeBegin=0,TimeEnd=0):
    """Extract the logs into a ufloat32 numpy array [index, ]"""

    '''
    global total_path
    sql = 'select count(distinct(path)) as total_path FROM PopularDB.FileFeature;'
    result = MyClient.query(sql)
    if len(result[1])>0:
        total_path = result[1][0][0]
        print(total_path)
    '''
    global path_td_feature_dict
    global total_path_dict
    """
    sql = 'select path,count(*) as count_num FROM PopularDB.FileFeature where Time>=FROM_UNIXTIME(%f) and ' \
          'Time<=FROM_UNIXTIME(%f) group by path order by count_num desc;' % (TimeBegin,TimeEnd)
    result = MyClient.query(sql)
    total_path_dict = {}
    for t in result[1]:
        if len(t)>0:
            total_path_dict[t[0]] = t[1]
    """

    td_time_dict = {}
    sql = "select distinct td FROM PopularDB.FileFeature where Time>=FROM_UNIXTIME(%f) " \
          "and Time<=FROM_UNIXTIME(%f);" % (TimeBegin, TimeEnd)
    td_result = MyClient.query(sql)
    for t in td_result[1]:
        sql = "select unix_timestamp(Time) FROM PopularDB.FileFeature where td='" + t[0].decode('unicode-escape')\
              + "' and Time>=FROM_UNIXTIME(%f) and Time<=FROM_UNIXTIME(%f) order by Time;" % (TimeBegin,TimeEnd)
        _result = MyClient.query(sql)
        begin = 0
        end = 0
        if len(_result[1])>0:
            begin = _result[1][0][0]
            end = _result[1][-1][0]
        td_time_dict[t[0].decode('unicode-escape')] = (begin,end)

    sql = "select distinct path,td FROM PopularDB.FileFeature where Time>=FROM_UNIXTIME(%f) " \
          "and Time<=FROM_UNIXTIME(%f);" % (TimeBegin,TimeEnd)
    distinct_result = MyClient.query(sql)

    if len(distinct_result[1])>0:
        for index in range(0,len(distinct_result[1])):
            _re = distinct_result[1][index]
            pathname = _re[0]
            print(pathname)
            td = _re[1]
            print(td)
            begin = td_time_dict[td.decode('unicode-escape')][0]
            HoursBefore = int((td_time_dict[td.decode('unicode-escape')][1]
                               -td_time_dict[td.decode('unicode-escape')][0])/3600)+1
            print HoursBefore
            path_td_feature_dict[pathname + td].append(HoursBefore)

            lognum_list = []
            for i in range(0, HoursBefore):
                lognum_list.append(0)
            '''
            with open('./file_feature/AccessNum/text/%s' % pathname.replace('/', '_'), 'a') as f:
                f.write("%s\n" % pathname)
            '''
            sql = "select unix_timestamp(Time) from FileFeature where path='%s' and td='%s' and Time>=FROM_UNIXTIME(%f) " \
                  "and TIME<=FROM_UNIXTIME(%f) order by Time;" % (pathname,td,TimeBegin,TimeEnd)
            time_series_result = MyClient.query(sql)
            for time in time_series_result[1]:
                lognum_list[int((time[0]-begin)/3600)] += 1
            print lognum_list
            '''
            """plot a figure about the file access num per hour."""
            fig = plt.figure(index)
            plt.plot(x_list, lognum_list, label='File Access Line')
            plt.xlabel('Hour Number')
            plt.ylabel('File Access Num')
            plt.title('File Access changes\nBy %d hour' % HoursBefore)
            plt.show()
            plt.savefig('./file_feature/AccessNum/png/%s.png' % pathname.replace('/', '_'))
            plt.close(index)
            #lognum_list = [0, 1, 0, 0.5, 0, 1, 0, 0.5, 0, 0, 1, 1, 0.5, 1, 0]
            '''

            #计算数学特征
            lognum_array = numpy.array(lognum_list)
            #计算总和
            totalNum = total_path_dict[pathname]
            print(totalNum)
            #path_td_feature_dict[pathname + td].append(totalNum)
            #计算均值
            mean = numpy.mean(lognum_array)
            print(mean)
            path_td_feature_dict[pathname + td].append(mean)
            #"""
            #计算中位数
            median = numpy.median(lognum_array)
            print(median)
            path_td_feature_dict[pathname + td].append(median)
            #计算众数
            #mode = scista.mode(lognum_list)
            #print(mode)
            #path_td_feature_dict[pathname + td].append(mode)
            #计算中列数
            midrange = Math.stats.midrange(lognum_list)
            print(midrange)
            path_td_feature_dict[pathname + td].append(midrange)
            #计算极差
            ptp = numpy.ptp(lognum_array)
            print(ptp)
            path_td_feature_dict[pathname + td].append(ptp)
            #计算方差
            var = numpy.var(lognum_array)
            print(var)
            path_td_feature_dict[pathname + td].append(var)
            #计算上四分位数
            UpQuater = 0
            try:
                UpQuater = Math.stats.quantile(lognum_list,p=0.25)
            except:
                UpQuater = 0
            print(UpQuater)
            #计算下四分位数
            DownQuater = 0
            try:
                DownQuater = Math.stats.quantile(lognum_list,p=0.75)
            except:
                DownQuater = 0
            print(DownQuater)
            #计算四分位差
            diff = DownQuater - UpQuater
            print(diff)
            path_td_feature_dict[pathname + td].append(diff)
            #计算变异系数
            variation = scista.variation(lognum_list)
            print(variation)
            path_td_feature_dict[pathname + td].append(variation)
            #计算自相关系数
            #at= Math.stats.autocorrelation(lognum_list)
            #print(at)
            #path_td_feature_dict[pathname + td].append(at)
            #计算信息熵
            ShannonEnt = Math.stats.calcShannonEnt(lognum_list)
            print(ShannonEnt)
            path_td_feature_dict[pathname + td].append(ShannonEnt)

            #计算小波系数
            #cA, cD = pywt.dwt(lognum_list,'db2')
            #print(cA)
            #print(cD)
            #"""
            """
            #计算傅立叶变换
            yy=scipy.fft(lognum_list)
            yf = abs(yy.real)
            yf1 = abs(yy.real)/len(lognum_list)
            yf2 = yf1[range(int(len(lognum_list)/2))]
            xf = numpy.arange(len(lognum_list))
            xf2 = xf[range(int(len(lognum_list)/2))]
            FrequencePeak = FindPeak(yf2)
            if len(FrequencePeak)>0:
                path_td_feature_dict[pathname + td].append(FrequencePeak[0][0])
                path_td_feature_dict[pathname + td].append(FrequencePeak[-1][0])
            else:
                path_td_feature_dict[pathname + td].append(-1)
                path_td_feature_dict[pathname + td].append(-1)
            '''
            plt.plot(xf2,yf2,'b')
            plt.show()
            '''
            """

            #最大连续为0时间长度
            ListZeroMax = listZeroLen(lognum_list)
            print(ListZeroMax)
            path_td_feature_dict[pathname + td].append(ListZeroMax)
            '''
            with open('./file_feature/AccessNum/text/%s' % pathname.replace('/', '_'), 'a') as f:
                f.write("%d  %d" % (23 - i, lognum))
                f.write("\n")
            '''
    pass


def extract_features01(TimeBegin=0,TimeEnd=0):
    """Extract the logs into a ufloat32 numpy array [index, ]"""
    global total_path_dict
    global path_td_feature_dict
    sql = 'select path,count(*) as count_num FROM PopularDB.FileFeature where Time>=FROM_UNIXTIME(%f) and ' \
          'Time<=FROM_UNIXTIME(%f) group by path order by count_num desc;' % (TimeBegin,TimeEnd)
    print sql
    result = MyClient.query(sql)
    total_path_dict = {}
    path_td_feature_dict = {}
    path_td_dict = {}
    for t in result[1]:
        if len(t)>1:
            total_path_dict[t[0]] = t[1]
            #path_feature_dict[t[0]] = []
    total_path_dict = OrderedDict(sorted(total_path_dict.items(),key= lambda t:t[1],reverse=True))

    sql = "select distinct path,td FROM PopularDB.FileFeature where Time>=FROM_UNIXTIME(%f) " \
          "and Time<=FROM_UNIXTIME(%f);" % (TimeBegin,TimeEnd)
    distinct_result = MyClient.query(sql)
    for t in distinct_result[1]:
        if t[0] not in path_td_dict.keys():
            path_td_dict[t[0]]=[]
            path_td_dict[t[0]].append(t[1])
        else:
            path_td_dict[t[0]].append(t[1])

    sql = "SELECT path,td,count(*),isuser,alpha,digit,alpha01,digit01,alpha02,digit02,path_depth,sum(OpenTime)," \
          "sum(rb/csize),sum(rb_min/csize),sum(rb_max/csize),sum(rb_sigma/csize),sum(wb/csize)," \
          "sum(wb_min/csize),sum(wb_max/csize),sum(wb_sigma/csize),sum(sfwdb/csize),sum(sbwdb/csize)," \
          "sum(sxlfwdb/csize),sum(sxlbwdb/csize),sum(nrc),sum(nwc),sum(nfwds),sum(nbwds),sum(nxlfwds)," \
          "sum(nxlbwds),sum(rt),sum(wt),csize FROM PopularDB.FileFeature where Time>=FROM_UNIXTIME(%f) and " \
          "Time<=FROM_UNIXTIME(%f) group by path,td;" % (TimeBegin,TimeEnd)
    #12 5 5 7 4
    feature_result = MyClient.query(sql)
    index = -1
    for t in feature_result[1]:
        if len(t)==33:
            index += 1
            _key = t[0]+t[1]
            if _key not in path_td_feature_dict.keys():
                path_td_feature_dict[t[0]+t[1]] = []
            #path_td_feature_dict[t[0] + t[1]].append(index)
            #path_td_feature_dict[t[0] + t[1]].append(t[2])
            path_td_feature_dict[t[0] + t[1]].append(t[3])
            path_td_feature_dict[t[0] + t[1]].append(t[11])
            path_td_feature_dict[t[0] + t[1]].append(t[12])
            path_td_feature_dict[t[0] + t[1]].append(t[13])
            path_td_feature_dict[t[0] + t[1]].append(t[14])
            path_td_feature_dict[t[0] + t[1]].append(t[15])
            path_td_feature_dict[t[0] + t[1]].append(t[16])
            path_td_feature_dict[t[0] + t[1]].append(t[17])
            path_td_feature_dict[t[0] + t[1]].append(t[18])
            path_td_feature_dict[t[0] + t[1]].append(t[19])
            path_td_feature_dict[t[0] + t[1]].append(t[20])
            path_td_feature_dict[t[0] + t[1]].append(t[21])
            path_td_feature_dict[t[0] + t[1]].append(t[22])
            path_td_feature_dict[t[0] + t[1]].append(t[23])
            path_td_feature_dict[t[0] + t[1]].append(t[24])
            path_td_feature_dict[t[0] + t[1]].append(t[25])
            path_td_feature_dict[t[0] + t[1]].append(t[26])
            path_td_feature_dict[t[0] + t[1]].append(t[27])
            path_td_feature_dict[t[0] + t[1]].append(t[28])
            path_td_feature_dict[t[0] + t[1]].append(t[29])
            path_td_feature_dict[t[0] + t[1]].append(t[30])
            path_td_feature_dict[t[0] + t[1]].append(t[31])
            path_td_feature_dict[t[0] + t[1]].append(t[32])
            '''
            '''



    #nfeature = numpy.array(path_td_feature_dict[0])
    #nfeature = numpy.array([],dtype=float)
    '''
    _list = []
    for k in path_td_feature_dict.keys():
        _list.append(path_td_feature_dict[k][0])
    nfeature = numpy.array(_list,dtype=float)
    '''
    feature_list = []
    '''
    for k in path_td_feature_dict.keys():
        _list.append(path_td_feature_dict[k])
    nfeature = numpy.array(_list,dtype=float)
    '''
    '''
    for t in range(0,len(path_td_feature_dict.values()[0])):
        #d = nfeature.shape
        _list = []
        for k in path_td_feature_dict.keys():
            _list.append(path_td_feature_dict[k][t])
        feature_list.append(_list)
    '''
    #featurelist_to_list(feature_list,0)
    #global feature_shape_list
    #feature_shape_array = numpy.array(feature_shape_list)
    #a = OrderedDict(sorted(path_td_feature_dict.items(), key=lambda d: d[1][0]))
    #b =
    #feature_shape_array = numpy.array([value[-1:] for key,value in a])

        #_numpy = numpy.array(_list)
        #nfeature = numpy.stack((nfeature,_numpy))
    #print nfeature

    #print nfeature.shape
    #return nfeature
    pass



def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(hot_file_percent=0.5):
    """extract labels."""
    global total_path_dict
    global path_td_feature_dict
    #a = sorted(path_td_feature_dict.items(), key=lambda d:d[1][0])
    #_keys_list = []
    #for item in a:
        #_keys_list.append(item[0])
    _keys_list = path_td_feature_dict.keys()
    label_list = []
    for t in _keys_list:
        label_list.append(0)
    FileNum = len(total_path_dict.keys())
    HotFileNum = int(FileNum * hot_file_percent)
    for index in range(0,FileNum-1):
        if index<HotFileNum:
            i = -1
            for t in _keys_list:
                i += 1
                if t.encode('unicode-escape').startswith(total_path_dict.keys()[index]):
                    label_list[i] = 1
        else:
            continue
    return label_list


    pass


class DataSet(object):

    def __init__(self,features,labels,one_hot=False,zoom=False,MaxValue=0,dtype=dtypes.float32):
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, excepted uint8 or float32' % dtype)
        if dtype == dtypes.float32:
            features = features.astype(numpy.float32)
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

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start==0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
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
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._features = self.features[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            features_new_part = self._features[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate((features_rest_part,features_new_part), axis=0), \
                   numpy.concatenate((labels_rest_part,labels_new_part),axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._features[start:end], self.labels[start:end]
        pass


def read_data_sets(train_time_begin, train_time_end, test_time_begin, test_time_end, validation_size=0,
                   num_classes=2, one_hot=False,dtype=dtypes.float32):
    """read data sets from mariadb."""
    global MyClient
    global total_path
    global total_path_dict
    global path_td_feature_dict
    global feature_shape_list
    global train_feature_shape_array
    global train_label_shape_array
    global test_feature_shape_array
    global test_label_shape_array
    MyClient = MysqlCon.MysqlCon(**MysqlConfig)

    # extract_features
    extract_features01(train_time_begin, train_time_end)
    extract_features(train_time_begin, train_time_end)
    train_feature_shape_array = numpy.array(path_td_feature_dict.values())
    print train_feature_shape_array.shape
    # extract_labels
    train_label_shape_array = numpy.array(extract_labels())
    if one_hot:
        train_label_shape_array = dense_to_one_hot(train_label_shape_array, num_classes)
    print train_label_shape_array

    # clear global variable
    total_path = 0
    total_path_dict = {}
    path_td_feature_dict = OrderedDict()
    feature_shape_list = []

    # extract_features
    extract_features01(test_time_begin, test_time_end)
    extract_features(test_time_begin, test_time_end)
    test_feature_shape_array = numpy.array(path_td_feature_dict.values())
    print test_feature_shape_array.shape
    # extract_labels
    test_label_shape_array = numpy.array(extract_labels())
    if one_hot:
        test_label_shape_array = dense_to_one_hot(test_label_shape_array, num_classes)
    print test_label_shape_array

    print "train_feature is:",train_feature_shape_array
    print "train_label is:",train_label_shape_array
    print "test_feature is:",test_feature_shape_array
    print "test_labels is:",test_label_shape_array

    validation_feature_shape_array = train_feature_shape_array[:validation_size]
    validation_label_shape_array = train_label_shape_array[:validation_size]
    t1 = train_feature_shape_array[validation_size:]
    t2 = train_label_shape_array[validation_size:]

    options = dict(dtype=dtype)

    train = DataSet(train_feature_shape_array,train_label_shape_array,**options)
    valiadation = DataSet(validation_feature_shape_array,validation_label_shape_array,**options)
    test = DataSet(test_feature_shape_array,test_label_shape_array,**options)
    MyClient.close()

    DataSets = []
    DataSets.append(t1)
    DataSets.append(t2)
    DataSets.append(test_feature_shape_array)
    DataSets.append(test_label_shape_array)
    #return Datasets(train=train,validation=valiadation,test=test)
    return DataSets
    pass


def main():
    read_data_sets(1510675200,1510718400,1510718400,1510761600)
    pass

if __name__ == '__main__':
    main()
