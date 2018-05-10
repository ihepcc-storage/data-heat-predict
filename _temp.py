# -*-coding:UTF-8-*-

import sys, os, traceback
import time
import numpy
import threading
import happybase
import numpy as np
import csv
from collections import OrderedDict
from tensorflow.tensorflow.python.framework import dtypes



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


class DataSet(object):

    def __init__(self,features,labels,train_time_begin,
                 train_time_end, PredictTimeOffset=0.0, PredictTimeWindow=0.0, csv01=None, csv02=None, csv03=None, csv04=None,
                 StartKey=None, ForceBalance=False,
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
        #self.StartKey = StartKey
        #self.OtsEnd = 9999999999.99
        self.Finish = 1

        self.arrayLock = threading.RLock()
        self.threads = []
        t = threading.Thread(target=self.read_data_set_series,args = (train_time_begin, train_time_end, PredictTimeOffset,
                                                              PredictTimeWindow, StartKey, csv01, csv02, csv03, csv04, ForceBalance))
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


    def read_data_set_series(self, train_time_begin, train_time_end, PredictTimeOffset=0.0, PredictTimeWindow=0.0,StartKey=None,
                    csv01=None, csv02=None, csv03=None, csv04=None, ForceBalance=False):
        if train_time_end <= train_time_begin:
            print 'Train time set error!', train_time_begin, train_time_end
            return -1


        OtsEnd = train_time_end
        ct = 0
        FeatrueList = []
        LableList = []

        #StartKey = None
        #ct = 0
        while True:
            a = time.time()

            #if self._features.shape[0] > self.capacity:
                #continue

            print 'StartKey', StartKey
            print 'ct', ct

            StartKey, ct = ScanHbaseTimeSeris(TimeBegin=train_time_begin, TimeEnd=OtsEnd, row_start=StartKey,
                                                            PredictTimeOffset=PredictTimeOffset,
                                                            PredictTimeWindow=PredictTimeWindow,csv01=csv01,
                                                                    csv02=csv02, csv03=csv03, csv04=csv04, limit=self.limit,
                                                                    ForceBalance=ForceBalance)
            #print train_time_begin, OtsEnd, ct
            """
            if train_time_begin > OtsEnd or self.OtsEnd <= OtsEnd:
                self.Finish = 0
                break
            else:
                self.OtsEnd = OtsEnd
            """
            if ct == 0:
                break
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


def ScanHbaseTimeSeris(TimeBegin=0.0, TimeEnd=0.0, row_start=None, PredictTimeOffset=0.0, PredictTimeWindow=0.0, limit=100, csv01=None,
                       csv02=None, csv03=None, csv04=None, ForceBalance=False):
    FeatureList = []
    LabelList = []
    ReadTimeDict = {}
    WriteTimeDict = {}
    global FilenameHashSet

    # global FilenameCacheList
    # global FilenameCacheListLen
    global LastAcessTimeTrainDict
    global AceessTimeDict
    global csvfileindex01, csvfileindex02, csvfileindex03, csvfileindex04
    #global filename01, filename02, filename03, filename04
    global CsvFeatureFile01, CsvFeatureFile02, CsvFeatureFile03, CsvFeatureFile04
    global filenamecsv
    looptoken = 3

    step = int((TimeEnd-TimeBegin)/3600)
    row_s = str(TimeBegin) + str('00000') + str('0') * 32
    if row_start!=None:
        row_s = row_start
    row_e = str(TimeEnd) + str('99999') + str('z') * 32
    ct = 0
    ct01, ct02, ct03, ct04 = 0, 0, 0, 0
    allct01, allct02, allct03, allct04 = 0, 0, 0, 0
    Ots = 0
    _HbaseTableKey = None
    try:
        for key, data in OtsUidPathTable.scan(row_start=row_s, row_stop=row_e):
            # print ct
            _HbaseTableKey = key
            ct += 1
            if ct>10000:
                break

            if ct01 > limit - 1:
                ct01 = 0
                CloseCsvFile(CsvFeatureFile01)
                csvfileindex01 += 1
                CsvFeatureFile01, csv01 = OpenCsvFile(filename01+'.'+str(csvfileindex01))
                #break

            if ct02 > limit - 1:
                ct02 = 0
                CloseCsvFile(CsvFeatureFile02)
                csvfileindex02 += 1
                CsvFeatureFile02, csv02 = OpenCsvFile(filename02+'.'+str(csvfileindex02))
                #break

            """
            if ct03 > limit - 1:
                ct03 = 0
                CloseCsvFile(CsvFeatureFile03)
                csvfileindex03 += 1
                CsvFeatureFile03, csv03 = OpenCsvFile(filename03+'.'+str(csvfileindex03))
                #break
            """

            if ct04 > limit - 1:
                ct04 = 0
                CloseCsvFile(CsvFeatureFile04)
                csvfileindex04 += 1
                CsvFeatureFile04, csv04 = OpenCsvFile(filename04+'.'+str(csvfileindex04))
                #break


            SingleFeatureList = []
            filenameHash = key[-32:]
            ruid = key[-37:-32]
            Ots = float(key[:-37])
            if filenameHash in FilenameHashSet:
                continue

            FilenameHashSet.add(filenameHash)

            try:
                ReadTime = ReadTimeDict[filenameHash]
                WriteTime = WriteTimeDict[filenameHash]
            except:
                ReadTimeDict[filenameHash] = 0
                WriteTimeDict[filenameHash] = 0

                TestTimeEnd = TimeBegin - PredictTimeOffset
                TestTimeBegin = TestTimeEnd - PredictTimeWindow

                row_st = filenameHash + str(TestTimeBegin) + '00000'
                row_ed = filenameHash + str(TestTimeEnd) + '99999'
                HighThreshold = sys.maxint
                count = 0
                for k, d in PathOtsUidTable.scan(row_start=row_st,
                                             row_stop=row_ed):
                    try:
                        # a little modification
                        log = d['cf:log']
                        log_list = log.split('&')
                        log_list.sort()
                        nrc = float(log_list[10].split('=')[1])
                        string = log_list[10]
                        nwc = float(log_list[11].split('=')[1])
                        string = log_list[11]
                        ReadTimeDict[filenameHash] += nrc
                        WriteTimeDict[filenameHash] += nwc
                    except:
                        print '0 hit', d
                    count += 1
                    if count > HighThreshold:
                        break
                    if ReadTimeDict[filenameHash]>0 and WriteTimeDict[filenameHash]>0:
                        break

                ReadTime = ReadTimeDict[filenameHash]
                WriteTime = WriteTimeDict[filenameHash]


            r_start = filenameHash + str(TimeBegin) + '00000'
            r_stop = filenameHash + str(TimeEnd) + '99999'
            nread = {}
            read_bytes = {}
            nwrite = {}
            write_bytes = {}
            osize = 0
            nseek = {}
            seek_bytes = {}
            SumAccessNum = 0
            a = time.time()
            for _key, _data in PathOtsUidTable.scan(row_start=r_start, row_stop=r_stop):
                SumAccessNum += 1
                Ots = float(_key[32:-5])
                TimeIndex = int((TimeEnd-Ots)/3600)
                try:
                    log = _data['cf:log']
                    log_list = log.split('&')
                    log_dict = {}
                    log_list.sort()
                    if log_list[0] == "":
                        del log_list[0]
                    for item in log_list:
                        ItemList = item.split('=')
                        if len(ItemList)>1:
                            log_dict[ItemList[0]] = ItemList[1]
                except Exception as e:
                    print e
                    continue
                try:
                    nread[TimeIndex] += int(log_dict['nrc'])
                except:
                    nread[TimeIndex] = int(log_dict['nrc'])
                try:
                    nwrite[TimeIndex] += int(log_dict['nwc'])
                except:
                    nwrite[TimeIndex] = int(log_dict['nwc'])
                try:
                    read_bytes[TimeIndex] += float(log_dict['rb'])
                except:
                    read_bytes[TimeIndex] = float(log_dict['rb'])
                try:
                    write_bytes[TimeIndex] += float(log_dict['wb'])
                except:
                    write_bytes[TimeIndex] = float(log_dict['wb'])
                try:
                    nseek[TimeIndex] += float(log_dict['nfwds'])
                except:
                    nseek[TimeIndex] = float(log_dict['nfwds'])
                nseek[TimeIndex] += float(log_dict['nbwds'])
                nseek[TimeIndex] += float(log_dict['nxlfwds'])
                nseek[TimeIndex] += float(log_dict['nxlbwds'])
                try:
                    seek_bytes[TimeIndex] += float(log_dict['sfwdb'])
                except:
                    seek_bytes[TimeIndex] = float(log_dict['sfwdb'])
                seek_bytes[TimeIndex] += float(log_dict['sbwdb'])
                seek_bytes[TimeIndex] += float(log_dict['sxlfwdb'])
                seek_bytes[TimeIndex] += float(log_dict['sxlbwdb'])
                if SumAccessNum == 1:
                    osize = float(log_dict['osize'])
            for i in range(0, step):
                try:
                    _list = [nread[i], read_bytes[i], nwrite[i], write_bytes[i], osize, nseek[i], seek_bytes[i]]
                except:
                    _list = [0, 0, 0, 0, osize, 0, 0]
                SingleFeatureList.extend(_list)

            print SingleFeatureList
            b = time.time()
            print b-a
            if SumAccessNum<=1 or sum(SingleFeatureList)==0:
                continue




            if ReadTime > 0 and WriteTime == 0:
                #LabelList.append([1, 0, 0, 0])
                SingleFeatureList.extend([1, 0, 0])
                if not ForceBalance or (ForceBalance and allct01 <= (allct01+allct02+allct04)/3):
                    try:
                        csv01.writerow(SingleFeatureList)
                        CsvFeatureFile01.flush()
                    except:
                        CloseCsvFile(CsvFeatureFile01)
                        CsvFeatureFile01, csv01 = OpenCsvFile(filename01 + '.' + str(csvfileindex01))
                        csv01.writerow(SingleFeatureList)
                        CsvFeatureFile01.flush()
                    ct01 += 1
                    allct01 += 1
                """
                if looptoken==0:
                    ct01 += 1
                    csv01.writerow(SingleFeatureList)
                    looptoken = 1
                """
            elif ReadTime >= 0 and WriteTime > 0:
                #LabelList.append([0, 1, 0, 0])
                SingleFeatureList.extend([0, 1, 0])
                if not ForceBalance or (ForceBalance and allct02 <= (allct01+allct02+allct04)/3):
                    try:
                        csv02.writerow(SingleFeatureList)
                        CsvFeatureFile02.flush()
                    except:
                        CloseCsvFile(CsvFeatureFile02)
                        CsvFeatureFile02, csv02 = OpenCsvFile(filename02 + '.' + str(csvfileindex02))
                        csv02.writerow(SingleFeatureList)
                        CsvFeatureFile02.flush()
                    ct02 += 1
                    allct02 += 1

            elif ReadTime==0 and WriteTime>0:
                continue
                #LabelList.append([0, 0, 1, 0])
                SingleFeatureList.extend([0, 0, 1, 0])
                if not ForceBalance or (ForceBalance and allct03 <= (allct01+allct02+allct03+allct04)/4):
                    csv03.writerow(SingleFeatureList)
                    CsvFeatureFile03.flush()
                    ct03 += 1
                    allct03 += 1

            else:
                #LabelList.append([0, 0, 0, 1])
                SingleFeatureList.extend([0, 0, 1])
                if not ForceBalance or (ForceBalance and allct04 <= (allct01+allct02+allct04)/3):
                    try:
                        csv04.writerow(SingleFeatureList)
                        CsvFeatureFile04.flush()
                    except:
                        CloseCsvFile(CsvFeatureFile04)
                        CsvFeatureFile04, csv04 = OpenCsvFile(filename04 + '.' + str(csvfileindex04))
                        csv04.writerow(SingleFeatureList)
                        CsvFeatureFile04.flush()
                    ct04 += 1
                    allct04 += 1

            try:
                filenamecsv.writerow(filenameHash)
            except:
                FilenameDatasetfile, filenamecsv = OpenCsvFile(FilenameDataset)
                filenamecsv.writerow([filenameHash])
            #print SingleFeatureList
            # ct += 1
            # csv.writerow(SingleFeatureList)
    except:
        print traceback.print_exc()
        return _HbaseTableKey, ct
        #return (FeatureList, LabelList, Ots, ct)
    return _HbaseTableKey, ct
    pass

def OpenCsvFile(filename=None):
    #global CsvFeatureFile
    CsvFeature = None
    CsvFeatureFile = None
    try:
        CsvFeatureFile = open(filename, "w")
        CsvFeature = csv.writer(CsvFeatureFile)
    except Exception as e:
        print e

    return (CsvFeatureFile ,CsvFeature)
    pass

def CloseCsvFile(filename=None):

    filename.close()
    pass


def main():
    features = np.empty(shape=[0, 144*8])
    labels = np.empty(shape=[0, 4])
    """
    CsvFeature = None
    try:
        CsvFeatureFile = open("csv/LSTM_TEST_DATASET.csv", "w")
        CsvFeature = csv.writer(CsvFeatureFile)
    except Exception as e:
        print e
    """
    #global csvfileindex
    #global CsvFeature
    #global filename

    global csvfileindex01, csvfileindex02, csvfileindex03, csvfileindex04
    global filename01, filename02, filename03, filename04
    global CsvFeatureFile01, CsvFeatureFile02, CsvFeatureFile03, CsvFeatureFile04
    global filenamecsv

    csvfileindex01, csvfileindex02, csvfileindex03, csvfileindex04 = 0,0,0,0
    filename01 = "csv/LSTM_TRAIN_DATASET_class01.csv"
    filename02 = "csv/LSTM_TRAIN_DATASET_class02.csv"
    filename03 = "csv/LSTM_TRAIN_DATASET_class03.csv"
    filename04 = "csv/LSTM_TRAIN_DATASET_class03.csv"
    FilenameDataset = "csv/LSTM_TRAIN_FILENAME_DATASET.csv"


    CsvFeatureFile01, csv01 = OpenCsvFile(filename01+'.'+str(csvfileindex01))
    CsvFeatureFile02, csv02 = OpenCsvFile(filename02 + '.' + str(csvfileindex02))
    #CsvFeatureFile03, csv03 = OpenCsvFile(filename03 + '.' + str(csvfileindex03))
    CsvFeatureFile04, csv04 = OpenCsvFile(filename04 + '.' + str(csvfileindex04))
    FilenameDatasetfile, filenamecsv = OpenCsvFile(FilenameDataset)

    dataset = DataSet(features=features, labels=labels, train_time_begin=9999999999.99 - 1523808000.00,
                                     train_time_end=9999999999.99 - 1523203200.00, PredictTimeOffset=0,
                                     PredictTimeWindow=86400.00, csv01=csv01, csv02=csv02, csv03=None, csv04=csv04,
                                    StartKey=None, ForceBalance=False, capacity=10000, limit=10000)

if __name__ == '__main__':
    main()

    """
    import cProfile
    import pstats
    cProfile.run("main()",filename="result.out", sort="cumulative")

    p = pstats.Stats("result.out")
    p.strip_dirs().sort_stats("cumulative", "name").print_stats(0.5)
    """