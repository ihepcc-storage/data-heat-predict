# -*-coding:UTF-8-*-

import sys, os
import time
import numpy
import Queue
import happybase
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

def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    return time.strftime(format,time.localtime(value))


def ScanHbase(rowStart = None, TimeBegin='0.0', TimeEnd='0.0', FilenameFeatureDict = OrderedDict(),FilenameReadtimeDict = OrderedDict(),
              FilenameFirstReadDict = OrderedDict(),FilenameRuidDict = OrderedDict()):

    global tableIndexStart

    for key, data in tableIndex.scan(row_start=rowStart):
        tableIndexStart = key
        oldfilename = data['name:filename']
        filename = list(oldfilename.split('/')[-1])
        directory = oldfilename[:len(oldfilename)-len(filename)]
        #if len(filename)!=len('runrds-00000000-00000000000.000.root'):
            #continue
        index = 0
        """
        for Achar in filename:
            if Achar.isdigit():
                filename[index] = '0'
            index += 1
        filename = ''.join(filename)
        """
        #if filename != 'runrds-00000000-00000000000.000.root':
            #continue

        for _key, _data in PathOtsUidTable.scan(row_start=(key+str(TimeBegin)+'00000'),
                                                row_stop=(key+str(TimeEnd)+'99999')):
            try:
                FilenameReadtimeDict[oldfilename] += 1
            except:
                FilenameReadtimeDict[oldfilename] = 1

            try:
                FilenameRuidDict[oldfilename].add(_key[45-len(_key):])
            except:
                FilenameRuidDict[oldfilename] = set()
                FilenameRuidDict[oldfilename].add(_key[45-len(_key):])

            try:
                row = table.row(_data['cf:logid'])
                if row['size:osize']=='0' and float(row['size:csize'])>0:
                    FilenameFirstReadDict[oldfilename] = float(row['size:ots'])
            except:
                del FilenameReadtimeDict[oldfilename]
                del FilenameRuidDict[oldfilename]
                continue

        #FirstWriteTime = 0.0
        try:
            FirstWriteTime = FilenameFirstReadDict[oldfilename]
        except:
            if oldfilename in FilenameReadtimeDict.keys():
                del FilenameReadtimeDict[oldfilename]
            if oldfilename in FilenameRuidDict.keys():
                del FilenameRuidDict[oldfilename]
            continue
        TimeRefer = FirstWriteTime+3600.0



        ruid = FilenameRuidDict[oldfilename].pop()


        a = ruid+str(FirstWriteTime)+key
        b = ruid+str(TimeRefer)+'z'*32
        FilenameFeatureDict[oldfilename] = []

        #print 'oldfilename,', oldfilename
        cnt = -1
        for k, d in UidOtsPathTable.scan(row_start=(ruid+str(FirstWriteTime)+key),row_stop=(ruid+str(TimeRefer)+'z'*32),limit=FeatureAxisLimit):
            cnt += 1
            if cnt==0:
                continue

            row = tableIndex.row(k[-32:])
            #print 'Time', FirstWriteTime, k
            #print 'add Feature:', row
            row = table.row(d['cf:logid'])

            osize = float(row['size:osize'])
            csize = float(row['size:csize'])
            rt = float(row['cf_r:rt'])
            wt = float(row['cf_w:wt'])
            nrc = int(row['cf_r:nrc'])
            nwc = int(row['cf_w:nwc'])
            nfwds = int(row['seek:nfwds'])
            nbwds = int(row['seek:nbwds'])
            sfwdb = float(row['seek:sfwdb'])
            sbwdb = float(row['seek:sbwdb'])
            nxlfwds = float(row['seek:nxlfwds'])
            nxlbwds = float(row['seek:nxlbwds'])
            sxlfwdb = float(row['seek:sxlfwdb'])
            sxlbwdb = float(row['seek:sxlbwdb'])

            try:
                FilenameFeatureDict[oldfilename].append(int(ruid))
                FilenameFeatureDict[oldfilename].append(osize)
                FilenameFeatureDict[oldfilename].append(csize)
                FilenameFeatureDict[oldfilename].append(rt)
                FilenameFeatureDict[oldfilename].append(wt)
                FilenameFeatureDict[oldfilename].append(nrc)
                FilenameFeatureDict[oldfilename].append(nwc)
                FilenameFeatureDict[oldfilename].append(nfwds)
                FilenameFeatureDict[oldfilename].append(nbwds)
                FilenameFeatureDict[oldfilename].append(sfwdb)
                FilenameFeatureDict[oldfilename].append(sbwdb)
                FilenameFeatureDict[oldfilename].append(nxlfwds)
                FilenameFeatureDict[oldfilename].append(nxlbwds)
                FilenameFeatureDict[oldfilename].append(sxlfwdb)
                FilenameFeatureDict[oldfilename].append(sxlbwdb)
            except:
                FilenameFeatureDict[oldfilename] = [int(ruid),osize,csize,rt,wt,nrc,nwc,nfwds,nbwds,sfwdb,sbwdb,nxlfwds,nxlbwds,sxlfwdb,sxlbwdb]

        if len(FilenameFeatureDict[oldfilename])<15*(FeatureAxisLimit-1):
            for i in range(0,15*(FeatureAxisLimit-1)-len(FilenameFeatureDict[oldfilename])):
                FilenameFeatureDict[oldfilename].append(-1)
        print len(FilenameFeatureDict.keys())

        if len(FilenameFeatureDict.keys()) >= FeatureNumLimit:
            break

        continue
    return(FilenameFeatureDict, FilenameReadtimeDict,FilenameFirstReadDict,FilenameRuidDict)
    pass


    pass


def extract_features(TimeBegin=0.0, TimeEnd=0.0):
    TimeBegin = '%.2f' % TimeBegin
    TimeEnd = '%.2f' % TimeEnd

    FilenameFeatureDict = OrderedDict()
    FilenameReadtimeDict = OrderedDict()
    FilenameFirstReadDict = OrderedDict()
    FilenameRuidDict = OrderedDict()
    while True:
        try:
            FilenameFeatureDict, FilenameReadtimeDict, FilenameFirstReadDict, FilenameRuidDict = ScanHbase(rowStart=tableIndexStart,
                  TimeBegin=TimeBegin,TimeEnd=TimeEnd,FilenameFeatureDict=FilenameFeatureDict,
                   FilenameReadtimeDict=FilenameReadtimeDict,FilenameFirstReadDict=FilenameFirstReadDict,FilenameRuidDict=FilenameRuidDict)
        except Exception as e:
            print e

        if len(FilenameFeatureDict.keys()) >= FeatureNumLimit:
            break


    FilenameReadtimeSet = sorted(FilenameReadtimeDict.items(),lambda x,y:cmp(x[1], y[1]),reverse=True)
    #print "FilenameReadtimeDict", FilenameReadtimeDict
    #print "FilenameReadtimeSet", FilenameReadtimeSet
    label_dict = FilenameReadtimeDict
    i = 0
    for t in FilenameReadtimeSet:
        i += 1
        if i<=len(label_dict)*0.1:
            label_dict[t[0]] = [1, 0]
        else:
            label_dict[t[0]] = [0, 1]
    #print 'label_dict', label_dict
    label_list = label_dict.values()
    #print 'label_list', label_list
    print "FilenameFeatureDict", FilenameFeatureDict
    #print 'feature_list', FilenameFeatureDict.keys()
    return FilenameFeatureDict, label_list
    return -1

    '''
        #return -1
        for t_key, t_data in UidOtsPathTable.scan(row_start=(ruid+str(TimeBegin)+key),
                                                  row_stop=(ruid+str(TimeEnd)+'0'*32),reverse=True):
            cnt += 1
            if cnt==1:
                continue
            if cnt>100:
                break

            rt = float(t_data['cf_r:rt'])
            wt = float(t_data['cf_w:wt'])
            nrc = int(t_data['cf_r:nrc'])
            nwc = int(t_data['cf_w:nwc'])
            nfwds = int(t_data['seek:nfwds'])
            nbwds = int(t_data['seek:nbwds'])
            nxlfwds = int(t_data['seek:nxlfwds'])
            nxlbwds = int(t_data['seek:nxlbwds'])

            try:
                FilenameFeatureDict[oldfilename].append(rt)
                FilenameFeatureDict[oldfilename].append(wt)
                FilenameFeatureDict[oldfilename].append(nrc)
                FilenameFeatureDict[oldfilename].append(nwc)
                FilenameFeatureDict[oldfilename].append(nfwds)
                FilenameFeatureDict[oldfilename].append(nbwds)
                FilenameFeatureDict[oldfilename].append(nxlfwds)
                FilenameFeatureDict[oldfilename].append(nxlbwds)
            except:
                FilenameFeatureDict[oldfilename] = [rt,wt,nrc,nwc,nfwds,nbwds,nxlfwds,nxlbwds]


        """
        cnt = 0
        for t_key, t_data in tableIndex.scan():
            p = t_data['name:filename']
            if p==oldfilename:
                continue
            f = p.split('/')[-1]
            if p[:len(p)-len(f)] != directory:
                continue
            print 'near filename', p
            cnt += 1
            if cnt>10:
                break

            count = 0
            rt = 0
            nrc = 0
            wt = 0
            nwc = 0
            nfwds = 0
            nxlfwds = 0
            nbwds = 0
            nxlbwds = 0
            for _key, _data in table.scan(row_start=(t_key+str(TimeBegin)),row_stop=(t_key+str(TimeEnd))):
                #if uid != t_data['Id:uid']:
                    #continue
                print _key, _data
                count += 1
                rt += float(_data['cf_r:rt'])
                wt += float(_data['cf_w:wt'])
                nrc += int(_data['cf_r:nrc'])
                nwc += int(_data['cf_w:nwc'])
                nfwds += int(_data['seek:nfwds'])
                nbwds += int(_data['seek:nbwds'])
                nxlfwds += int(_data['seek:nxlfwds'])
                nxlbwds += int(_data['seek:nxlbwds'])

            try:
                FilenameFeatureDict[oldfilename].append(count)
                FilenameFeatureDict[oldfilename].append(rt)
                FilenameFeatureDict[oldfilename].append(wt)
                FilenameFeatureDict[oldfilename].append(nrc)
                FilenameFeatureDict[oldfilename].append(nwc)
                FilenameFeatureDict[oldfilename].append(nfwds)
                FilenameFeatureDict[oldfilename].append(nbwds)
                FilenameFeatureDict[oldfilename].append(nxlfwds)
                FilenameFeatureDict[oldfilename].append(nxlbwds)
            except:
                FilenameFeatureDict[oldfilename] = [count,rt,wt,nrc,nwc,nfwds,nbwds,nxlfwds,nxlbwds]

            print FilenameFeatureDict
        """



        '''
    pass

def read_data_sets(train_time_begin, train_time_end, test_time_begin, test_time_end, validation_size=10,
                   num_classes=2, one_hot=False,dtype=dtypes.float32):

    if train_time_end<=train_time_begin:
        print 'Train time set error!'
        return -1

    if test_time_end<=test_time_begin:
        print 'Test time set error!'
        return -1


    FilenameFeatureDict, label_list = extract_features(TimeBegin=train_time_begin,TimeEnd=train_time_end)
    train_feature_array = numpy.array(FilenameFeatureDict.values())
    print train_feature_array.shape
    train_label_array = numpy.array(label_list)
    print train_label_array.shape


    FilenameFeatureDict, label_list = extract_features(TimeBegin=test_time_begin, TimeEnd=test_time_end)
    test_feature_array = numpy.array(FilenameFeatureDict.values())
    print test_feature_array.shape
    test_label_array = numpy.array(label_list)
    print test_label_array.shape


    validation_feature_array = train_feature_array[:validation_size]
    validation_label_array = train_label_array[:validation_size]

    options = dict(dtype=dtype)

    train = DataSet(train_feature_array, train_label_array, **options)
    validation = DataSet(validation_feature_array, validation_label_array, **options)
    test = DataSet(test_feature_array, test_label_array, **options)
    DataSets = []
    DataSets.append(train)
    DataSets.append(validation)
    DataSets.append(test)
    return DataSets

    """
    DataSets = []
    DataSets.append(train_feature_array)
    DataSets.append(train_label_array)
    DataSets.append(validation_feature_array)
    DataSets.append(validation_label_array)
    return DataSets
    """

    pass

def main():
    #extract_features(TimeEnd=9999999999.99-1519833600.00,TimeBegin=9999999999.99-1560179200.00)
    DataSets = read_data_sets(train_time_end=9999999999.99 - 1520611200.00,
                                           train_time_begin=9999999999.99 - 1521043200.00,
                                           test_time_end=9999999999.99 - 1521129600.00,
                                           test_time_begin=9999999999.99 - 1521475200.00)
    pass

if __name__ == '__main__':
    main()