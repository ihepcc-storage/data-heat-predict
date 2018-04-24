# -*-coding:UTF-8-*-
import sys, os
import logging
import config
import time
from collections import OrderedDict
import collections
import multiprocessing
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


MysqlConfig = {
    'user':'chengzj',
    'passwd':'chengzj@2016',
    'host':'localhost',
    'db':'PopularDB'
}
MyClient = None
total_path = 0
total_path_dict = {}
path_td_feature_dict = OrderedDict()
feature_shape_list = []
feature_shape_array = None
label_shape_array = None

#Datasets = collections.namedtuple('Datasets',['train','validation','test'])

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


def extract_td(TimeBegin=0,TimeEnd=0):
    """"""
    sql = "select distinct td FROM PopularDB.FileFeature where Time>=FROM_UNIXTIME(%f) " \
          "and Time<=FROM_UNIXTIME(%f);" % (TimeBegin, TimeEnd)
    td_result = MyClient.query(sql)
    return td_result[1]


def extract_features(path_td_feature_dict,
                           TimeBegin=0,TimeEnd=0,path_td_dict={},MyClient=None):
    """Extract the logs into a ufloat32 numpy array [index, ]"""

    td_time_dict = {}
    path_list = path_td_dict.keys()
    for path in path_td_dict:
        for td in path_td_dict[path]:
            #td = t[1].decode('unidcode-escape')
            sql = "select unix_timestamp(Time) FROM PopularDB.FileFeature where td='" + td \
                + "' and Time>=FROM_UNIXTIME(%f) and Time<=FROM_UNIXTIME(%f) order by Time;" % (TimeBegin,TimeEnd)
            _result = MyClient.query(sql)
            begin = 0
            end = 0
            if len(_result[1])>0:
                begin = _result[1][0][0]
                end = _result[1][-1][0]
            td_time_dict[td] = (begin,end)

            HoursBefore = int((td_time_dict[td][1]
                               -td_time_dict[td][0])/3600)+1
            print HoursBefore
            if (path+td) not in path_td_feature_dict.keys():
                path_td_feature_dict[path + td] = []
            path_td_feature_dict[path + td].append(HoursBefore)

            lognum_list = []
            for i in range(0, HoursBefore):
                lognum_list.append(0)

            sql = "select unix_timestamp(Time) from FileFeature where path='%s' and td='%s' and Time>=FROM_UNIXTIME(%f) " \
                  "and TIME<=FROM_UNIXTIME(%f) order by Time;" % (path,td,TimeBegin,TimeEnd)
            time_series_result = MyClient.query(sql)
            for time in time_series_result[1]:
                lognum_list[int((time[0]-begin)/3600)] += 1
            print lognum_list


            #计算数学特征
            lognum_array = numpy.array(lognum_list)
            #计算总和
            #totalNum = total_path_dict[pathname]
            #print(totalNum)
            #path_td_feature_dict[pathname + td].append(totalNum)
            #计算均值
            mean = numpy.mean(lognum_array)
            print(mean)
            path_td_feature_dict[path + td].append(mean)
            #"""
            #计算中位数
            median = numpy.median(lognum_array)
            print(median)
            path_td_feature_dict[path + td].append(median)
            #计算众数
            #mode = scista.mode(lognum_list)
            #print(mode)
            #path_td_feature_dict[pathname + td].append(mode)
            #计算中列数
            midrange = Math.stats.midrange(lognum_list)
            print(midrange)
            path_td_feature_dict[path + td].append(midrange)
            #计算极差
            ptp = numpy.ptp(lognum_array)
            print(ptp)
            path_td_feature_dict[path + td].append(ptp)
            #计算方差
            var = numpy.var(lognum_array)
            print(var)
            path_td_feature_dict[path + td].append(var)
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
            path_td_feature_dict[path + td].append(diff)
            #计算变异系数
            variation = scista.variation(lognum_list)
            print(variation)
            path_td_feature_dict[path + td].append(variation)
            #计算自相关系数
            #at= Math.stats.autocorrelation(lognum_list)
            #print(at)
            #path_td_feature_dict[pathname + td].append(at)
            #计算信息熵
            ShannonEnt = Math.stats.calcShannonEnt(lognum_list)
            print(ShannonEnt)
            path_td_feature_dict[path + td].append(ShannonEnt)

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
            path_td_feature_dict[path + td].append(ListZeroMax)
    return path_td_feature_dict
    pass


def extract_features01(TimeBegin=0,TimeEnd=0,path_td_dict={},MyClient=None):
    """Extract the logs into a ufloat32 numpy array [index, ]"""

    total_path_dict = {}
    path_td_feature_dict = {}
    #path_td_dict = {}
    path_result = path_td_dict.keys()
    for path in path_result:
        #td = t.decode('unidcode-escape')
        sql = 'select path,count(*) as count_num FROM PopularDB.FileFeature where Time>=FROM_UNIXTIME(%f) and ' \
            'Time<=FROM_UNIXTIME(%f) and path = "%s" group by path order by count_num desc;' % (TimeBegin,TimeEnd,path)
        result = MyClient.query(sql)
        for re in result[1]:
            if len(re)>1:
                total_path_dict[re[0]] = re[1]

        sql = "SELECT td,count(*),isuser,alpha,digit,alpha01,digit01,alpha02,digit02,path_depth,sum(OpenTime)," \
              "sum(rb/csize),sum(rb_min/csize),sum(rb_max/csize),sum(rb_sigma/csize),sum(wb/csize)," \
              "sum(wb_min/csize),sum(wb_max/csize),sum(wb_sigma/csize),sum(sfwdb/csize),sum(sbwdb/csize)," \
              "sum(sxlfwdb/csize),sum(sxlbwdb/csize),sum(nrc),sum(nwc),sum(nfwds),sum(nbwds),sum(nxlfwds)," \
              "sum(nxlbwds),sum(rt),sum(wt),csize FROM PopularDB.FileFeature where Time>=FROM_UNIXTIME(%f) and " \
              "Time<=FROM_UNIXTIME(%f) and path = '%s' group by path,td;" % (TimeBegin,TimeEnd,path)
        feature_result = MyClient.query(sql)
        index = -1
        for re in feature_result[1]:
            if len(re)==32:
                index += 1
                _key = path+re[0].encode('unicode-escape')
                if _key not in path_td_feature_dict.keys():
                    path_td_feature_dict[_key] = []
                #path_td_feature_dict[_key].append(index)
                #path_td_feature_dict[_key].append(re[1])
                path_td_feature_dict[_key].append(re[1])
                path_td_feature_dict[_key].append(re[2])
                path_td_feature_dict[_key].append(re[10])
                path_td_feature_dict[_key].append(re[11])
                path_td_feature_dict[_key].append(re[12])
                path_td_feature_dict[_key].append(re[13])
                path_td_feature_dict[_key].append(re[14])
                path_td_feature_dict[_key].append(re[15])
                path_td_feature_dict[_key].append(re[16])
                path_td_feature_dict[_key].append(re[17])
                path_td_feature_dict[_key].append(re[18])
                path_td_feature_dict[_key].append(re[19])
                path_td_feature_dict[_key].append(re[20])
                path_td_feature_dict[_key].append(re[21])
                path_td_feature_dict[_key].append(re[22])
                path_td_feature_dict[_key].append(float(re[23]))
                path_td_feature_dict[_key].append(float(re[24]))
                path_td_feature_dict[_key].append(float(re[25]))
                path_td_feature_dict[_key].append(float(re[26]))
                path_td_feature_dict[_key].append(float(re[27]))
                path_td_feature_dict[_key].append(float(re[28]))
                path_td_feature_dict[_key].append(re[29])
                path_td_feature_dict[_key].append(re[30])
                path_td_feature_dict[_key].append(re[31])
    return (path_td_feature_dict,total_path_dict)
    pass



def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(path_td_feature_dict,total_path_dict,path_td_dict,hot_file_percent=0.2):
    """extract labels."""
    total_path_dict = OrderedDict(sorted(total_path_dict.items(),key= lambda t:t[1],reverse=True))
    label_dict = path_td_feature_dict
    FileNameList = total_path_dict.keys()
    FileNum = len(FileNameList)
    HotFileNum = int(FileNum * hot_file_percent)

    for index in range(0,FileNum):
        Filename = FileNameList[index]
        try:
            for td in path_td_dict[Filename]:
                if index<HotFileNum:
                    label_dict[Filename+td] = 1
                else:
                    label_dict[Filename + td] = 0
        except:
            print "Extract label Error!"
            return []
    print len(label_dict.values())
    return label_dict.values()
    pass

def func(TimeBegin=0,TimeEnd=0,path_td_dict={},MysqlConfig=None):
    MyClient = MysqlCon.MysqlCon(**MysqlConfig)
    (path_td_feature_dict,total_path_dict) = extract_features01(TimeBegin,TimeEnd,path_td_dict,MyClient)
    path_td_feature_dict = extract_features(path_td_feature_dict,TimeBegin,TimeEnd,path_td_dict,MyClient)
    #return path_td_feature_str+"|"+total_path_str
    MyClient.close()
    return(path_td_feature_dict,total_path_dict)
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
                   numpy.concatenate((features_rest_part,features_new_part),axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._features[start:end], self.labels[start:end]
        pass


def read_data_sets(time_begin=0,time_end=0,FileNameList=[], one_hot=False,dtype=dtypes.float32):
    """read data sets from mariadb."""

    path_td_dict = {}
    MyClient = MysqlCon.MysqlCon(**MysqlConfig)
    for filename in FileNameList:
        sql = "select distinct td from PopularDB.FileFeature where path='%s' and Time>=FROM_UNIXTIME(%f) and " \
              "Time<=FROM_UNIXTIME(%f);" % (filename,time_begin, time_end)
        result = MyClient.query(sql)
        for re in result[1]:
            if filename not in path_td_dict.keys():
                path_td_dict[filename] = []
                path_td_dict[filename].append(re[0])
            else:
                path_td_dict[filename].append(re[0])
    (path_td_feature_dict,total_path_dict) = func(time_begin,time_end,path_td_dict,MysqlConfig)

    train_feature_shape_array = numpy.array(path_td_feature_dict.values())
    print train_feature_shape_array.shape

    MyClient.close()

    DataSets = []
    DataSets.append(train_feature_shape_array)
    return DataSets
    pass


def main():
    read_data_sets(1511107200,1511193600,['/eos/user/x/xuhs/booster_TMCI_20170726_lat_xiy_0_Nonlinear_1p5mm/Results_15f0_16385bins_1000000MP/18.00nC/wp00_s.txt',
                                          '/eos/user/x/xuhs/booster_TMCI_20170726_lat_xiy_0_Nonlinear_1p5mm/Results_15f0_16385bins_1000000MP/18.00nC/mwitrack.twi'])
    #read_data_sets(1510675200, 1510718400, 1510718400, 1510761600)
    pass

if __name__ == '__main__':
    main()
