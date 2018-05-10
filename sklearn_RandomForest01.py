# -*-coding:UTF-8-*-

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import cross_validation, metrics

from sklearn.utils.validation import check_is_fitted
from sklearn.utils import column_or_1d
from sklearn.utils import shuffle

import matplotlib.pylab as plt

class NewLabelEncoder(LabelEncoder):

    def transform_diff(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : new labels list
        """
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        classes = np.unique(y)
        diff = np.setdiff1d(classes, self.classes_)
        return diff
        #    raise ValueError("y contains new labels: %s" % str(diff))



class RandomForest(RandomForestClassifier):

    """
    def __init__(self, train_csv='', test_csv=''):

        #self.train_data = train_data
        #self.test_data = test_data


        pass
    """


    def read_csv(self, train_csv1=None, train_csv2=None, test_csv1=None, test_csv2=None, ForceBalance=False, Shuffle=True):
        train_data1 = pd.read_csv(train_csv1, dtype='object')
        if ForceBalance:
            train_shape1 = train_data1.shape[0]
        else:
            train_shape1 = None
        train_data2 = pd.read_csv(train_csv2, dtype='object', nrows=train_shape1)
        self.train_data = pd.concat([train_data1, train_data2])
        if Shuffle:
            self.train_data = shuffle(self.train_data)

        self.train_data = self.train_data['ruid,rgid,filename_suffix,filename_depth,create_host,filename_part2,filename_part3,' \
                                          'filename_part4,filename_part5,filename_part6,filename_part7,filename_part8,' \
                                          'filename_part9,BatchFile'.split(',')]
        if self.train_data.isnull().values.any():
            print 'NaN value exist in dataframe train_data:', np.nonzero(self.train_data.isnull().any(1))[0].tolist()
            print self.train_data[self.train_data.isnull().any(1)]
            self.train_data = self.train_data.drop(np.nonzero(self.train_data.isnull().any(1))[0].tolist(), axis=0)


        test_data1 = pd.read_csv(test_csv1, dtype='object')
        if ForceBalance:
            test_shape1 = test_data1.shape[0]
        else:
            test_shape1 = None
        test_data2 = pd.read_csv(test_csv2, dtype='object', nrows=test_shape1)
        self.test_data = pd.concat([test_data1, test_data2])
        if Shuffle:
            self.test_data = shuffle(self.test_data)

        self.test_data = self.test_data['ruid,rgid,filename_suffix,filename_depth,create_host,filename_part2,filename_part3,' \
                                          'filename_part4,filename_part5,filename_part6,filename_part7,filename_part8,' \
                                          'filename_part9,BatchFile'.split(',')]
        if self.test_data.isnull().values.any():
            print 'NaN value exist in dataframe test_data:', np.nonzero(self.test_data.isnull().any(1))[0].tolist()
            print self.test_data[self.test_data.isnull().any(1)]
            self.test_data = self.test_data.drop(np.nonzero(self.test_data.isnull().any(1))[0].tolist(), axis=0)
        pass


    def BuildEncoder(self, DataFrame = None):
        self.ruid_encoder = NewLabelEncoder()
        self.ruid_encoder.fit(DataFrame['ruid'])

        self.rgid_encoder = NewLabelEncoder()
        self.rgid_encoder.fit(DataFrame['rgid'])

        self.filename_suffix_encoder = NewLabelEncoder()
        self.filename_suffix_encoder.fit(DataFrame['filename_suffix'])

        self.filename_depth_encoder = NewLabelEncoder()
        self.filename_depth_encoder.fit(DataFrame['filename_depth'])

        self.create_host_encoder = NewLabelEncoder()
        self.create_host_encoder.fit(DataFrame['create_host'])

        self.filename_part2_encoder = NewLabelEncoder()
        self.filename_part2_encoder.fit(DataFrame['filename_part2'])

        self.filename_part3_encoder = NewLabelEncoder()
        self.filename_part3_encoder.fit(DataFrame['filename_part3'])

        self.filename_part4_encoder = NewLabelEncoder()
        self.filename_part4_encoder.fit(DataFrame['filename_part4'])

        self.filename_part5_encoder = NewLabelEncoder()
        self.filename_part5_encoder.fit(DataFrame['filename_part5'])

        self.filename_part6_encoder = NewLabelEncoder()
        self.filename_part6_encoder.fit(DataFrame['filename_part6'])

        self.filename_part7_encoder = NewLabelEncoder()
        self.filename_part7_encoder.fit(DataFrame['filename_part7'])

        self.filename_part8_encoder = NewLabelEncoder()
        self.filename_part8_encoder.fit(DataFrame['filename_part8'])

        self.filename_part9_encoder = NewLabelEncoder()
        self.filename_part9_encoder.fit(DataFrame['filename_part9'])

        self.BatchFile_encoder = NewLabelEncoder()
        self.BatchFile_encoder.fit(DataFrame['BatchFile'])

        pass


    def Encode(self, DataFrame=None):

        for column in 'ruid,rgid,filename_suffix,filename_depth,create_host,filename_part2,filename_part3,' \
                                          'filename_part4,filename_part5,filename_part6,filename_part7,filename_part8,' \
                                          'filename_part9,BatchFile'.split(','):
            encoder = None
            if column=='ruid':
                encoder = self.ruid_encoder
            if column=='rgid':
                encoder = self.rgid_encoder
            if column=='filename_suffix':
                encoder = self.filename_suffix_encoder
            if column=='filename_depth':
                encoder = self.filename_depth_encoder
            if column=='create_host':
                encoder = self.create_host_encoder
            if column=='filename_part2':
                encoder = self.filename_part2_encoder
            if column=='filename_part3':
                encoder = self.filename_part3_encoder
            if column=='filename_part4':
                encoder = self.filename_part4_encoder
            if column=='filename_part5':
                encoder = self.filename_part5_encoder
            if column=='filename_part6':
                encoder = self.filename_part6_encoder
            if column=='filename_part7':
                encoder = self.filename_part7_encoder
            if column=='filename_part8':
                encoder = self.filename_part8_encoder
            if column=='filename_part9':
                encoder = self.filename_part9_encoder
            if column=='BatchFile':
                encoder = self.BatchFile_encoder

            print 'Encode:', column
            returncode = -1
            while(returncode!=0):
                try:
                    DataFrame[column] = encoder.transform(DataFrame[column])
                    returncode = 0
                except ValueError, e:
                    #NewLabelList = e.message.split(': ')[1][1:-1].replace("'",'').replace("\n",'').split(' ')
                    NewLabelList = encoder.transform_diff(DataFrame[column])
                    #print e
                    #print NewLabelList
                    DataFrame[column] = DataFrame[column].replace(NewLabelList, 'none')
                    try:
                        DataFrame[column] = encoder.transform(DataFrame[column])
                        returncode = 0
                    except ValueError, e:
                        print e
                        returncode = -1
        return DataFrame
        pass


    def fitNew(self, n_estimators=100):
        self.x_train = self.train_data.values[1::, 0:13]
        self.y_train = self.train_data.values[1::, 13].astype(np.int)
        # print self.y_train, self.y_train.tolist().count(1), self.y_train.tolist().count(0)

        #self.classfier = RandomForestClassifier(n_estimators=n_estimators)
        #self.classfier.fit(x_train, y_train)
        self.fit(self.x_train, self.y_train)
        #print 'oob_score_', self.oob_score


        pass

    def scoreNew(self):
        self.x_test = self.test_data.values[1::, 0:13]
        self.y_test = self.test_data.values[1::, 13].astype(np.int)
        #print y_test, y_test.tolist().count(1), y_test.tolist().count(0)

        #scores = self.classfier.score(x_test, y_test)
        scores = self.score(self.x_test, self.y_test)
        print 'Test Dataset Scores:', scores

        #y_predprob = self.predict_proba(self.x_test)
        #print "AUC Score (Train): %f" % metrics.roc_auc_score(self.y_test, y_predprob)

def main():
    rf = RandomForest(n_estimators=100, max_features=1.0)
    rf.read_csv("csv/feature_batch.csv", "csv/feature_user.csv", "csv/test_batch.csv", "csv/test_user.csv",
                Shuffle=True)
    rf.BuildEncoder(rf.train_data)
    rf.train_data = rf.Encode(rf.train_data)
    print rf.train_data['BatchFile'].value_counts()

    rf.test_data = rf.Encode(rf.test_data)
    print rf.test_data['BatchFile'].value_counts()

    rf.fitNew()
    print 'feature importance:', rf.feature_importances_
    scores = cross_val_score(rf, rf.x_train, rf.y_train, cv=10)
    print 'cross_val_score:', scores


    rf.scoreNew()

if __name__ == '__main__':
    main()