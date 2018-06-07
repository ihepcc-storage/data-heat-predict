# -*-coding:UTF-8-*-

import sys, os, traceback
import time
import numpy
import happybase
import numpy as np
import csv
import threading
import copy
import math
from collections import OrderedDict
from scipy import sparse


# from tensorflow.python.framework import dtypes

ct01List, ct02List, ct03List, ct04List, ct05List = [],[],[],[],[]
TotalDimension = 4327



def openHbase():
    global connection, table, tableIndex, UidOtsPathTable, UidPathOtsTable, PathOtsUidTable, OtsUidPathTable
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
FilenameHashSet = set()

LastAcessTimeTrainDict = {}
AceessTimeDict = {}
#ReadTimeList, WriteTimeList = [], []

MatrixFeatureList01,MatrixFeatureList02,MatrixFeatureList03,MatrixFeatureList04 = [],[],[],[]



class DataSet(object):
    def __init__(self, features, labels, train_time_begin,
                 train_time_end, PredictTimeOffsetList=None, PredictTimeWindowList=None, csv01=None, csv02=None, csv03=None,
                 csv04=None,csv05=None,
                 StartKey=None, ForceBalance=False,
                 limit=100, capacity=10000, one_hot=False, zoom=False, MaxValue=0, dtype=numpy.float32):
        # dtype = dtypes.as_dtype(dtype).base_dtype
        # if dtype not in (dtypes.uint8, dtypes.float32):
        # raise TypeError('Invalid image dtype %r, excepted uint8 or float32' % dtype)
        # if dtype == dtypes.float32:
        #features = features.astype(numpy.float32)
        #if zoom:
            #features = features.multiply(features, 1.0 / MaxValue)
        self.capacity = capacity
        self.limit = limit
        self._features = features
        #self._num_examples = features.shape[0]
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        # self.StartKey = StartKey
        # self.OtsEnd = 9999999999.99
        self.Finish = 1

        self.arrayLock = threading.RLock()
        self.threads = []
        t = threading.Thread(target=self.read_data_set_series,
                             args=(train_time_begin, train_time_end, PredictTimeOffsetList,
                                   PredictTimeWindowList, StartKey, csv01, csv02, csv03, csv04, ForceBalance))
        self.threads.append(t)
        t.start()

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
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._features = self.features[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
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
            return numpy.concatenate((features_rest_part, features_new_part), axis=0), \
                   numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._features[start:end], self.labels[start:end]
        pass

    def read_data_set_series(self, train_time_begin, train_time_end, PredictTimeOffsetList=None, PredictTimeWindowList=None,
                             StartKey=None,
                             csv01=None, csv02=None, csv03=None, csv04=None, csv05=None, ForceBalance=False):

        #global ReadTimeList, WriteTimeList
        if train_time_end <= train_time_begin:
            print 'Train time set error!', train_time_begin, train_time_end
            return -1

        OtsEnd = train_time_end
        ct = 0

        while True:
            a = time.time()

            # if self._features.shape[0] > self.capacity:
            # continue

            print 'StartKey', StartKey
            print 'ct', ct

            StartKey, ct = ScanHbaseTimeSeris(TimeBegin=train_time_begin, TimeEnd=OtsEnd, row_start=StartKey,
                                              PredictTimeOffsetList=PredictTimeOffsetList,
                                              PredictTimeWindowList=PredictTimeWindowList, csv01=csv01,
                                              csv02=csv02, csv03=csv03, csv04=csv04, csv05=csv05, limit=self.limit,
                                              ForceBalance=ForceBalance)
            # print train_time_begin, OtsEnd, ct
            """
            if train_time_begin > OtsEnd or self.OtsEnd <= OtsEnd:
                self.Finish = 0
                break
            else:
                self.OtsEnd = OtsEnd
            """
            if ct == 1 and StartKey != None:
                break
        pass

    def next_batch_series(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        # Shuffle for the first epoch
        # if self._epochs_completed == 0 and start==0:
        # perm0 = numpy.arange(self.features.shape[0])
        # self._features = self.features[perm0]
        # self._labels = self.labels[perm0]
        # Go to the next epoch
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
    return time.strftime(format, time.localtime(value))


def ScanHbaseTimeSeris(TimeBegin=0.0, TimeEnd=0.0, row_start=None, PredictTimeOffsetList=None, PredictTimeWindowList=None,
                       limit=100, csv01=None,
                       csv02=None, csv03=None, csv04=None, csv05=None, ForceBalance=False):
    #ReadTimeDict = {}
    #WriteTimeDict = {}

    #global ReadTimeList, WriteTimeList
    global FilenameHashSet
    global LastAcessTimeTrainDict
    global AceessTimeDict
    global npzfileindex01List, npzfileindex02List, npzfileindex03List, npzfileindex04List, npzfileindex05List
    global MatrixFeatureList01,MatrixFeatureList02,MatrixFeatureList03,MatrixFeatureList04
    global dirList
    global filenamecsv, FilenameDataset

    step = int((TimeEnd - TimeBegin) / 3600)

    row_s = str(TimeBegin) + str('00000') + str('0') * 32
    if row_start != None:
        row_s = row_start
    row_e = str(TimeEnd) + str('99999') + str('z') * 32
    ct = 0

    global ct01List, ct02List, ct03List, ct04List, ct05List
    allct01, allct02, allct03, allct04, allct05 = 0, 0, 0, 0, 0
    _HbaseTableKey = None

    try:
        for key, data in OtsUidPathTable.scan(row_start=row_s, row_stop=row_e):
            # print ct
            _HbaseTableKey = key
            ct += 1
            if ct > 5000:
                break

            print 'time:', timestamp_datetime(
                9999999999.99 - float(key[:-37])), 'ct01:', ct01List, 'ct02:', ct02List, 'ct03:', ct03List, 'ct04:', ct04List, 'ct05:', ct05List
            for index in range(0, len(ct01List)):
                ct01 = ct01List[index]
                if ct01 > limit - 1:
                    ct01List[index] = 0
                    npzfileindex01List[index] += 1
                    sparse.save_npz(dirList[index]+filename01+'.'+str(npzfileindex01List[index])+'.npz', MatrixFeatureList01[index])
                    MatrixFeatureList01[index] = sparse.coo_matrix((1, TotalDimension))

                ct02 = ct02List[index]
                if ct02 > limit - 1:
                    ct02List[index] = 0
                    npzfileindex02List[index] += 1
                    sparse.save_npz(dirList[index] + filename02 + '.' + str(npzfileindex02List[index]) + '.npz',
                                    MatrixFeatureList02[index])
                    MatrixFeatureList02[index] = sparse.coo_matrix((1, TotalDimension))

                ct03 = ct03List[index]
                if ct03 > limit - 1:
                    ct03List[index] = 0
                    npzfileindex03List[index] += 1
                    sparse.save_npz(dirList[index] + filename03 + '.' + str(npzfileindex03List[index]) + '.npz',
                                    MatrixFeatureList03[index])
                    MatrixFeatureList03[index] = sparse.coo_matrix((1, TotalDimension))

                ct04 = ct04List[index]
                if ct04 > limit - 1:
                    ct04List[index] = 0
                    npzfileindex04List[index] += 1
                    sparse.save_npz(dirList[index] + filename04 + '.' + str(npzfileindex04List[index]) + '.npz',
                                    MatrixFeatureList04[index])
                    MatrixFeatureList04[index] = sparse.coo_matrix((1, TotalDimension))


            SingleFeatureList = []
            filenameHash = key[-32:]
            if filenameHash in FilenameHashSet:
                continue

            FilenameHashSet.add(filenameHash)

            r_start = filenameHash + str(TimeBegin) + '00000'
            r_stop = filenameHash + str(TimeEnd) + '99999'
            nread = {}
            read_bytes = {}
            nwrite = {}
            write_bytes = {}
            csize = 0
            nseek = {}
            seek_bytes = {}
            SumAccessNum = 0
            for i in range(0, step):
                nread[i],nwrite[i],read_bytes[i],write_bytes[i],nseek[i],seek_bytes[i] = 0,0,0,0,0,0
            for _key, _data in PathOtsUidTable.scan(row_start=r_start, row_stop=r_stop):

                Ots = float(_key[32:-5])
                TimeIndex = int((TimeEnd - Ots) / 3600)
                try:
                    log = _data['cf:log']
                    log_list = log.split('&')
                    log_dict = {}
                    log_list.sort()
                    if log_list[0] == "":
                        del log_list[0]
                    for item in log_list:
                        ItemList = item.split('=')
                        if len(ItemList) > 1:
                            log_dict[ItemList[0]] = ItemList[1]
                except Exception as e:
                    print e
                    continue

                if int(log_dict['nrc'])>0:
                    nread[TimeIndex] += 1
                    SumAccessNum += 1

                if int(log_dict['nwc'])>0:
                    nwrite[TimeIndex] += 1
                    SumAccessNum += 1


                read_bytes[TimeIndex] += float(log_dict['rb'])

                write_bytes[TimeIndex] += float(log_dict['wb'])

                nseek[TimeIndex] += float(log_dict['nfwds'])
                nseek[TimeIndex] += float(log_dict['nbwds'])
                nseek[TimeIndex] += float(log_dict['nxlfwds'])
                nseek[TimeIndex] += float(log_dict['nxlbwds'])

                seek_bytes[TimeIndex] += float(log_dict['sfwdb'])
                seek_bytes[TimeIndex] += float(log_dict['sbwdb'])
                seek_bytes[TimeIndex] += float(log_dict['sxlfwdb'])
                seek_bytes[TimeIndex] += float(log_dict['sxlbwdb'])
                if float(log_dict['csize']) > csize:
                    csize = float(log_dict['csize'])
            for i in range(0, step):
                try:
                    _list = [nread[i], read_bytes[i], nwrite[i], write_bytes[i], nseek[i], seek_bytes[i]]
                except Exception as e:
                    print 'keyError!',i
                    print nread.keys()
                    _list = [0, 0, 0, 0, 0, 0]
                SingleFeatureList.extend(_list)
            SingleFeatureList.append(csize)
            if SumAccessNum == 0 or sum(SingleFeatureList[:-1]) == 0:
                continue

            ReadTimeList, WriteTimeList = [], []
            for index in range(0, len(PredictTimeOffsetList)):
                ReadTimeList.append(0)
                WriteTimeList.append(0)

                PredictTimeOffset = PredictTimeOffsetList[index]
                PredictTimeWindow = PredictTimeWindowList[index]
                TestTimeEnd = TimeBegin - PredictTimeOffset
                TestTimeBegin = TestTimeEnd - PredictTimeWindow

                row_st = filenameHash + str(TestTimeBegin) + '00000'
                row_ed = filenameHash + str(TestTimeEnd) + '99999'
                HighThreshold = 10001
                count = 0
                for k, d in PathOtsUidTable.scan(row_start=row_st,
                                                 row_stop=row_ed):
                    try:
                        # a little modification
                        log = d['cf:log']
                        log_list = log.split('&')
                        log_dict = {}
                        log_list.sort()
                        if log_list[0] == "":
                            del log_list[0]
                        for item in log_list:
                            ItemList = item.split('=')
                            if len(ItemList) > 1:
                                log_dict[ItemList[0]] = ItemList[1]

                        nrc = int(log_dict['nrc'])
                        nwc = int(log_dict['nwc'])
                        ReadTimeList[index] += nrc
                        WriteTimeList[index] += nwc
                    except:
                        print '0 hit', d
                    count += 1
                    if count > HighThreshold:
                        break
                    if k[:32] != filenameHash:
                        break

            for index in range(0, len(ReadTimeList)):
                _SingleFeatureList = copy.deepcopy(SingleFeatureList)
                ReadTime = ReadTimeList[index]
                WriteTime = WriteTimeList[index]

                if ReadTime>=0 and ReadTime<=10:
                    if WriteTime>0:
                        _SingleFeatureList.extend([1,0,0,0,1,0])
                    else:
                        _SingleFeatureList.extend([1,0,0,0,0,1])
                    SingleFeatureMatrix = sparse.coo_matrix([_SingleFeatureList])
                    MatrixFeatureList01[index] = sparse.vstack((MatrixFeatureList01[index], SingleFeatureMatrix))
                    ct01List[index] += 1

                elif ReadTime > 10 and ReadTime <= 100:
                    if WriteTime > 0:
                        _SingleFeatureList.extend([0, 1, 0, 0, 1, 0])
                    else:
                        _SingleFeatureList.extend([0, 1, 0, 0, 0, 1])
                    SingleFeatureMatrix = sparse.coo_matrix([_SingleFeatureList])
                    MatrixFeatureList02[index] = sparse.vstack((MatrixFeatureList02[index], SingleFeatureMatrix))
                    ct02List[index] += 1

                elif ReadTime > 100 and ReadTime <= 1000:
                    if WriteTime > 0:
                        _SingleFeatureList.extend([0, 0, 1, 0, 1, 0])
                    else:
                        _SingleFeatureList.extend([0, 0, 1, 0, 0, 1])
                    SingleFeatureMatrix = sparse.coo_matrix([_SingleFeatureList])
                    MatrixFeatureList03[index] = sparse.vstack((MatrixFeatureList03[index], SingleFeatureMatrix))
                    ct03List[index] += 1

                elif ReadTime > 1000:
                    if WriteTime > 0:
                        _SingleFeatureList.extend([0, 0, 0, 1, 1, 0])
                    else:
                        _SingleFeatureList.extend([0, 0, 0, 1, 0, 1])
                    SingleFeatureMatrix = sparse.coo_matrix([_SingleFeatureList])
                    MatrixFeatureList04[index] = sparse.vstack((MatrixFeatureList04[index], SingleFeatureMatrix))
                    ct04List[index] += 1
                #print _SingleFeatureList

            """
            try:
                filenamecsv.writerow([filenameHash])
                FilenameDatasetfile.writestr(FilenameDataset + '.zip', string_buffer05.getvalue())
            except:
                FilenameDatasetfile, filenamecsv = OpenCsvFile(FilenameDataset + '.zip', string_buffer05)
                filenamecsv.writerow([filenameHash])
                FilenameDatasetfile.writestr(FilenameDataset + '.zip', string_buffer05.getvalue())
            """

    except:
        print 'error!'
        traceback.print_exc()
        a = traceback.print_exc()
        if str(a).find('sendall') > 0:
            print 'Broken pipe'
        print traceback.print_exc()
        openHbase()
        return _HbaseTableKey, ct
        # return (FeatureList, LabelList, Ots, ct)

    """
    CsvFeatureFile01.close()
    CsvFeatureFile02.close()
    CsvFeatureFile03.close()
    CsvFeatureFile04.close()
    CsvFeatureFile05.close()
    FilenameDatasetfile.close()
    """
    return _HbaseTableKey, ct
    pass


def OpenCsvFile(filename=None):
    # global CsvFeatureFile
    CsvFeature = None
    CsvFeatureFile = None
    try:
        # CsvFeatureFile = open(filename, "w")
        CsvFeatureFile = open(filename, 'a')
        CsvFeature = csv.writer(CsvFeatureFile)
    except Exception as e:
        print e

    return (CsvFeatureFile, CsvFeature)
    pass


def CloseCsvFile(filename=None):
    filename.close()
    pass


def main():
    global npzfileindex01List, npzfileindex02List, npzfileindex03List, npzfileindex04List, npzfileindex05List
    global filename01, filename02, filename03, filename04, filename05, FilenameDataset
    global dirList

    global filenamecsv
    global ct01List, ct02List, ct03List, ct04List, ct05List
    global MatrixFeatureList01,MatrixFeatureList02,MatrixFeatureList03,MatrixFeatureList04

    npzfileindex01List, npzfileindex02List, npzfileindex03List, npzfileindex04List, npzfileindex05List = [],[],[],[],[]

    #  4/1  1522512000
    #  5/1  1525104000

    # 1525795200  2018/5/9 0:0:0
    # 1525190400  2018/5/2 0:0:0

    # 1526400000  2018/5/16 0:0:0
    # 1525795200  2018/5/9 0:0:0

    time_end = 1525104000.00
    time_begin = 1522512000.00

    filename01 = "LSTM_TRAIN_"+str(int(time_begin))+"-"+str(int(time_end))+"_DATASET_class01"
    filename02 = "LSTM_TRAIN_"+str(int(time_begin))+"-"+str(int(time_end))+"_DATASET_class02"
    filename03 = "LSTM_TRAIN_"+str(int(time_begin))+"-"+str(int(time_end))+"_DATASET_class03"
    filename04 = "LSTM_TRAIN_"+str(int(time_begin))+"-"+str(int(time_end))+"_DATASET_class04"
    filename05 = "LSTM_TRAIN_"+str(int(time_begin))+"-"+str(int(time_end))+"_DATASET_class05"
    FilenameDataset = "LSTM_TRAIN_FILENAME_DATASET.csv"

    dirList = ["csv01/","csv02/","csv03/"]
    PredictTimeOffsetList, PredictTimeWindowList = [0,0,0], [86400.00*3,86400.00*7,86400.00*14]

    for index in range(0, len(PredictTimeWindowList)):
        ct01List.append(0)
        ct02List.append(0)
        ct03List.append(0)
        ct04List.append(0)
        ct05List.append(0)
        npzfileindex01List.append(0)
        npzfileindex02List.append(0)
        npzfileindex03List.append(0)
        npzfileindex04List.append(0)
        npzfileindex05List.append(0)
        MatrixFeatureList01.append(sparse.coo_matrix((1, TotalDimension)))
        MatrixFeatureList02.append(sparse.coo_matrix((1, TotalDimension)))
        MatrixFeatureList03.append(sparse.coo_matrix((1, TotalDimension)))
        MatrixFeatureList04.append(sparse.coo_matrix((1, TotalDimension)))



    dataset = DataSet(features=None,labels=None, train_time_begin=9999999999.99 - 1525104000.00,
                      train_time_end=9999999999.99 - 1522512000.00, PredictTimeOffsetList=PredictTimeOffsetList,
                      PredictTimeWindowList=PredictTimeWindowList,
                      StartKey=None, ForceBalance=False, capacity=10000, limit=5000)



    for index in range(0, len(ct01List)):
        ct01 = ct01List[index]
        if ct01 > 0:
            ct01List[index] = 0
            npzfileindex01List[index] += 1
            sparse.save_npz(dirList[index] + filename01 + '.' + str(npzfileindex01List[index]) + '.npz',
                            MatrixFeatureList01[index])
            MatrixFeatureList01[index] = sparse.coo_matrix((1, TotalDimension))

        ct02 = ct02List[index]
        if ct02 > 0:
            ct02List[index] = 0
            npzfileindex02List[index] += 1
            sparse.save_npz(dirList[index] + filename02 + '.' + str(npzfileindex02List[index]) + '.npz',
                            MatrixFeatureList02[index])
            MatrixFeatureList02[index] = sparse.coo_matrix((1, TotalDimension))

        ct03 = ct03List[index]
        if ct03 > 0:
            ct03List[index] = 0
            npzfileindex03List[index] += 1
            sparse.save_npz(dirList[index] + filename03 + '.' + str(npzfileindex03List[index]) + '.npz',
                            MatrixFeatureList03[index])
            MatrixFeatureList03[index] = sparse.coo_matrix((1, TotalDimension))

        ct04 = ct04List[index]
        if ct04 > 0:
            ct04List[index] = 0
            npzfileindex04List[index] += 1
            sparse.save_npz(dirList[index] + filename04 + '.' + str(npzfileindex04List[index]) + '.npz',
                            MatrixFeatureList04[index])
            MatrixFeatureList04[index] = sparse.coo_matrix((1, TotalDimension))


if __name__ == '__main__':
    openHbase()
    main()

    """
    import cProfile
    import pstats
    cProfile.run("main()",filename="result.out", sort="cumulative")

    p = pstats.Stats("result.out")
    p.strip_dirs().sort_stats("cumulative", "name").print_stats(0.5)
    """
