# -*-coding:UTF-8-*-

import sys, os
import time
import numpy
import threading
import happybase
import numpy as np
import csv
from collections import OrderedDict
from tensorflow.python.framework import dtypes



connection = happybase.Connection('192.168.60.64', autoconnect=True)
table = connection.table('eos_log')
tableIndex = connection.table('eos_log_index')
UidOtsPathTable = connection.table("eoslog_UidOtsPathMap")
UidPathOtsTable = connection.table("eoslog_UidPathOtsMap")
PathOtsUidTable = connection.table("eoslog_PathOtsUidMap")
OtsUidPathTable = connection.table("eoslog_OtsUidPathMap")

tableIndexStart = None
FeatureNumLimit = 100
FeatureAxisLimit = 101


LastAcessTimeTrainDict = {}
AceessTimeDict = {}


class DataSet(object):

    def __init__(self,features,labels,train_time_begin,
                 train_time_end, PredictTimeOffset=0.0, PredictTimeWindow=0.0, n_steps = 1000, CsvTestDataset=None,
                   limit=100, capacity=10000, one_hot=False,zoom=False,MaxValue=0,dtype=dtypes.float32):
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, excepted uint8 or float32' % dtype)
        if dtype == dtypes.float32:
            features = features.astype(numpy.float32)
        if zoom:
            features = features.multiply(features, 1.0/MaxValue)
        self.capacity = capacity
        self.limit = limit
        self._features = features
        self._num_examples = features.shape[0]
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.OtsEnd = 9999999999.99
        self.Finish = 1

        self.arrayLock = threading.RLock()
        self.threads = []
        t = threading.Thread(target=self.read_data_set_series,args = (train_time_begin, train_time_end, PredictTimeOffset,
                                                              PredictTimeWindow, n_steps, CsvTestDataset))
        self.threads.append(t)
        t.start()
        t.join()



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


    def read_data_set_series(self, train_time_begin, train_time_end, PredictTimeOffset=0.0, PredictTimeWindow=0.0,
                   n_steps = 1000, csv=None):
        if train_time_end <= train_time_begin:
            print 'Train time set error!'
            return -1


        OtsEnd = train_time_end
        ct = 0
        FeatrueList = []
        LableList = []


        while True:
            a = time.time()

            if self._features.shape[0] > self.capacity:
                continue

            print train_time_begin, OtsEnd

            FeatrueList, LableList, OtsEnd, ct = ScanHbaseTimeSeris(TimeBegin=train_time_begin, TimeEnd=OtsEnd,
                                                            PredictTimeOffset=PredictTimeOffset,
                                                            PredictTimeWindow=PredictTimeWindow,csv=csv, n_steps=n_steps, limit=self.limit)
            print train_time_begin, OtsEnd, ct
            if train_time_begin > OtsEnd or self.OtsEnd <= OtsEnd:
                self.Finish = 0
                break
            else:
                self.OtsEnd = OtsEnd

            """
            train_feature_array = np.array(FeatrueList)
            #print train_feature_array.shape
            if train_feature_array.shape[0]==0:
                continue
            #print train_feature_array.shape
            train_label_array = np.array(LableList)
            #print train_label_array.shape

            self.arrayLock.acquire()
            self._features = np.concatenate((self._features, train_feature_array),axis=0)
            self._labels = np.concatenate((self._labels, train_label_array), axis=0)
            self.arrayLock.release()
            print self._features.shape
            print self._labels.shape
            
            b = time.time()
            print 'time', b-a
            """
        pass

    def next_batch_series(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        # Shuffle for the first epoch
        #if self._epochs_completed == 0 and start==0:
            #perm0 = numpy.arange(self.features.shape[0])
            #self._features = self.features[perm0]
            #self._labels = self.labels[perm0]
        #Go to the next epoch
        # print 'fetch a batch'
        if self.Finish == 0:
            if batch_size > self._features.shape[0]:
                print 'end of dataset'
                return np.array([-1]), np.array([-1])

        if batch_size > self._features.shape[0]:
            time.sleep(10)
            print 'not enough featrue', self._features.shape[0]
            return self.next_batch_series(batch_size)
        else:
            print 'a new batch'
            self.arrayLock.acquire()
            if shuffle:
                perm = numpy.arange(self._features.shape[0])
                numpy.random.shuffle(perm)
                self._features = self._features[perm]
                self._labels = self._labels[perm]

            self._features_slice = np.split(self._features, [batch_size, self._features.shape[0]], axis=0)
            self._features = self._features_slice[1]
            self._labels_slice = np.split(self._labels, [batch_size, self._labels.shape[0]], axis=0)
            self._labels = self._labels_slice[1]
            self.arrayLock.release()
            return (self._features_slice[0], self._labels_slice[0])
        pass


def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    return time.strftime(format,time.localtime(value))


def ScanHbaseTimeSeris(TimeBegin=0.0, TimeEnd=0.0, row_s=None, PredictTimeOffset=0.0, PredictTimeWindow=0.0, n_steps=1000, limit=100, csv=None):
    FeatureList = []
    LabelList = []
    ReadTimeDict = {}
    WriteTimeDict = {}

    # global FilenameCacheList
    # global FilenameCacheListLen
    global LastAcessTimeTrainDict
    global AceessTimeDict

    step = int((TimeBegin-TimeEnd)/3600)
    row_s = str(TimeBegin) + str('00000') + str('0') * 32
    row_e = str(TimeEnd) + str('99999') + str('z') * 32
    ct = 0
    Ots = 0
    for key, data in OtsUidPathTable.scan(row_start=row_e,
                                          row_stop=row_s, reverse=True):
        # print ct
        if ct > limit - 1:
            break
        ct += 1

        SingleFeatureList = []
        filenameHash = key[-32:]
        ruid = key[-37:-32]
        Ots = float(key[:-37])
        """
        try:
            t = len(FilenameCacheList[ruid])
        except:
            FilenameCacheList[ruid] = []
        """
        try:
            AceessTimeDict[filenameHash] += 1
            AccessTime = AceessTimeDict[filenameHash]
        except:
            AceessTimeDict[filenameHash] = 1
            AccessTime = 1

        try:
            LastTrainAccessTime = LastAcessTimeTrainDict[filenameHash]
        except:
            LastAcessTimeTrainDict[filenameHash] = AccessTime
            LastTrainAccessTime = AccessTime

        if (AccessTime - LastTrainAccessTime) < n_steps / 5:
            continue
        LastAcessTimeTrainDict[filenameHash] = AccessTime



        r_start = key[-32:] + key[:-37] + key[-37:-32]
        for _key, _data in PathOtsUidTable.scan(row_start=r_start, limit=n_steps):
            try:
                if _key[:32] != filenameHash:
                    break

                ots, cts, sfwdb, sbwdb, sxlbwdb, sxlfwdb = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                rb, rb_min, rb_max, rb_sigma, wb, wb_min, wb_max, wb_sigma = 0, 0, 0, 0, 0, 0, 0, 0
                nrc, nwc, nbwds, nfwds, nxlbwds, nxlfwds, osize, csize = 0, 0, 0, 0, 0, 0, 0, 0
                rt, wt = 0.0, 0.0
                log = _data['cf:log']
                log_list = log.split('&')
                ruid = log_list[2].split('=')[1]
                ots = float(log_list[9].split('=')[1])
                otms = float(log_list[10].split('=')[1])
                ots += otms / 1000.0
                ots = '%.2f' % ots
                cts = float(log_list[11].split('=')[1])
                ctms = float(log_list[12].split('=')[1])
                cts += ctms / 1000.0
                cts = '%.2f' % cts
                rb = float(log_list[13].split('=')[1])
                rb_min = float(log_list[14].split('=')[1])
                rb_max = float(log_list[15].split('=')[1])
                rb_sigma = float(log_list[16].split('=')[1])
                wb = float(log_list[17].split('=')[1])
                wb_min = float(log_list[18].split('=')[1])
                wb_max = float(log_list[19].split('=')[1])
                wb_sigma = float(log_list[21].split('=')[1])
                sfwdb = float(log_list[22].split('=')[1])
                sbwdb = float(log_list[23].split('=')[1])
                sxlfwdb = float(log_list[24].split('=')[1])
                sxlbwdb = float(log_list[25].split('=')[1])
                nrc = float(log_list[26].split('=')[1])
                nwc = float(log_list[27].split('=')[1])
                nfwds = float(log_list[28].split('=')[1])
                nbwds = float(log_list[29].split('=')[1])
                nxlfwds = float(log_list[30].split('=')[1])
                nxlbwds = float(log_list[31].split('=')[1])
                rt = float(log_list[32].split('=')[1])
                wt = float(log_list[33].split('=')[1])
                osize = float(log_list[34].split('=')[1])
                csize = float(log_list[35].split('=')[1])
                _list = [ots, cts, osize, csize, rb, rb_min, rb_max, rb_sigma, wb, wb_min,
                         wb_max, wb_sigma, rt, wt, nrc, nwc, nfwds, nbwds, sfwdb, sbwdb, nxlfwds,
                         nxlbwds, sxlfwdb, sxlbwdb]
                _list.reverse()
                SingleFeatureList.extend(_list)

            except:
                continue
        if len(SingleFeatureList) < 24 * n_steps:
            for i in range(0, 24 * n_steps - len(SingleFeatureList)):
                SingleFeatureList.append(-1)
        SingleFeatureList.reverse()
        #FeatureList.append(SingleFeatureList)

        ReadTime = 0
        WriteTime = 0

        try:
            ReadTime = ReadTimeDict[filenameHash]
            WriteTime = WriteTimeDict[filenameHash]
        except:
            ReadTimeDict[filenameHash] = 0
            WriteTimeDict[filenameHash] = 0

            TestTimeEnd = Ots - PredictTimeOffset
            TestTimeBegin = TestTimeEnd - PredictTimeWindow

            row_st = filenameHash + str(TestTimeBegin) + '00000'
            row_ed = filenameHash + str(TestTimeEnd) + '99999'
            HighThreshold = 1
            count = 0
            for k, d in PathOtsUidTable.scan(row_start=row_st,
                                             row_stop=row_ed):
                try:
                    # a little modification
                    """
                    _r = table.row(d['cf:logid'])
                    nrc = int(_r['cf_r:nrc'])
                    nwc = int(_r['cf_w:nwc'])
                    """
                    log = d['cf:log']
                    log_list = log.split('&')
                    nrc = float(log_list[26].split('=')[1])
                    nwc = float(log_list[27].split('=')[1])
                    ReadTimeDict[filenameHash] += nrc
                    WriteTimeDict[filenameHash] += nwc
                except:
                    print '0 hit', d
                    _r = table.row(d['cf:logid'])
                    nrc = int(_r['cf_r:nrc'])
                    nwc = int(_r['cf_w:nwc'])
                    ReadTimeDict[filenameHash] += nrc
                    WriteTimeDict[filenameHash] += nwc
                count += 1
                if count > HighThreshold:
                    break

            ReadTime = ReadTimeDict[filenameHash]
            WriteTime = WriteTimeDict[filenameHash]


        if ReadTime > 0 and WriteTime == 0:
            #LabelList.append([1, 0, 0, 0])
            SingleFeatureList.extend([1, 0, 0, 0])
        elif ReadTime > 0 and WriteTime > 0:
            #LabelList.append([0, 1, 0, 0])
            SingleFeatureList.extend([0, 1, 0, 0])
        elif ReadTime==0 and WriteTime>0:
            #LabelList.append([0, 0, 1, 0])
            SingleFeatureList.extend([0, 0, 1, 0])
        else:
            #LabelList.append([0, 0, 0, 1])
            SingleFeatureList.extend([0, 0, 0, 1])

        print SingleFeatureList
        csv.writerow(SingleFeatureList)

    return (FeatureList, LabelList, Ots, ct)
    pass


def read_data_sets(train_time_begin, train_time_end, test_time_begin, test_time_end, PredictTimeOffset=0.0, PredictTimeWindow=0.0,
                   validation_size=10, num_classes=3, one_hot=False,dtype=dtypes.float32):

    if train_time_end<=train_time_begin:
        print 'Train time set error!'
        return -1

    if test_time_end<=test_time_begin:
        print 'Test time set error!'
        return -1

    FeatrueList, LableList, Ots = ScanHbaseTimeSeris(TimeBegin=train_time_begin,TimeEnd=train_time_end,
                                                PredictTimeOffset=PredictTimeOffset ,PredictTimeWindow=PredictTimeWindow,
                                                     n_steps=nsteps, limit=200)
    train_feature_array = np.array(FeatrueList)
    print train_feature_array.shape
    train_label_array = np.array(LableList)
    print train_label_array.shape

    options = dict(dtype=dtype)

    train = DataSet(train_feature_array, train_label_array, **options)

    DataSets = []
    DataSets.append(train)
    return DataSets
    pass

def main():
    """
    #DataSets = read_data_sets(train_time_end=9999999999.99 - 1521521100.00,
                                           train_time_begin=9999999999.99 - 1521553500.00,
                                           test_time_end=9999999999.99 - 1521129600.00,
                                           test_time_begin=9999999999.99 - 1521475200.00,PredictTimeOffset=86400.00,
                                            PredictTimeWindow=86400.00
                              )
    #dataset0 = DataSets[0]
    #features, lables = dataset0.next_batch(batch_size=16)
    """
    features = np.empty(shape=[0, 2400])
    labels = np.empty(shape=[0, 4])
    CsvTestDataset = None
    CsvFeature = None
    try:
        CsvFeatureFile = open("csv/LSTM_TEST_DATASET.csv", "w")
        CsvFeature = csv.writer(CsvFeatureFile)
    except Exception as e:
        print e

    CsvFeature.writerow("ots, cts, osize, csize, rb, rb_min, rb_max, rb_sigma, wb, wb_min,\
                         wb_max, wb_sigma, rt, wt, nrc, nwc, nfwds, nbwds, sfwdb, sbwdb, nxlfwds,\
                         nxlbwds, sxlfwdb, sxlbwdb, read, read-write, write, NoAccess".split(','))

    dataset = DataSet(features=features, labels=labels, train_time_begin=9999999999.99 - 1521993600.00,
                                     train_time_end=9999999999.99 - 1521475200.00, PredictTimeOffset=86400.00,
                                     PredictTimeWindow=86400.00,n_steps=100, CsvTestDataset=CsvFeature,
                                     capacity=10000, limit=1000)

if __name__ == '__main__':
    main()

    """
    import cProfile
    import pstats
    cProfile.run("main()",filename="result.out", sort="cumulative")

    p = pstats.Stats("result.out")
    p.strip_dirs().sort_stats("cumulative", "name").print_stats(0.5)
    """