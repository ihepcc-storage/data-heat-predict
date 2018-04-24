# -*-coding:UTF-8-*-

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
import matplotlib.pylab as plt

train_data = pd.read_csv("csv/feature.csv", dtype='object')

train_data = train_data['ruid,rgid,filename_suffix,filename_depth,filename_part2,filename_part3,filename_part4,' \
                        'filename_part5,filename_part6,filename_part7,filename_part8,filename_part9,BatchFile'.split(',')]


#train_data.dropna()
if train_data.isnull().values.any():
    print 'NaN value exist in dataframe train_data'
    print np.nonzero(train_data.isnull().any(1))[0].tolist()
    print train_data[train_data.isnull().any(1)]
    train_data = train_data.drop(np.nonzero(train_data.isnull().any(1))[0].tolist(),axis=0)

#text = 'ruid,rgid,filename_suffix,filename_depth,filename_part2,filename_part3,filename_part4,filename_part5, \
#         filename_part6,filename_part7,filename_part8,filename_part9,BatchFile'.split(',')

#train_data = pd.DataFrame(train_data.values[0::, 0::], columns=text)

s_ruid = train_data['ruid']
s_ruid = LabelEncoder().fit(s_ruid).transform(s_ruid)
train_data['ruid'] = s_ruid

s_gid = train_data['rgid']
s_gid = LabelEncoder().fit(s_gid).transform(s_gid)
train_data['rgid'] = s_gid

s_filenamesuffix = train_data['filename_suffix']
s_filenamesuffix = LabelEncoder().fit(s_filenamesuffix).transform(s_filenamesuffix)
train_data['filename_suffix'] = s_filenamesuffix

s_filenamedepth = train_data['filename_depth']
s_filenamedepth = LabelEncoder().fit(s_filenamedepth).transform(s_filenamedepth)
train_data['filename_depth'] = s_filenamedepth

s_filename_part = train_data['filename_part2']
s_filename_part = LabelEncoder().fit(s_filename_part).transform(s_filename_part)
train_data['filename_part2'] = s_filename_part

s_filename_part = train_data['filename_part3']
# _t = train_data.loc[train_data.filename_part3=='nan']
"""
le = LabelEncoder()
le.fit(s_filename_part)
print le.classes_
"""
s_filename_part = LabelEncoder().fit(s_filename_part).transform(s_filename_part)
train_data['filename_part3'] = s_filename_part

s_filename_part = train_data['filename_part4']
s_filename_part = LabelEncoder().fit(s_filename_part).transform(s_filename_part)
train_data['filename_part4'] = s_filename_part

s_filename_part = train_data['filename_part5']
s_filename_part = LabelEncoder().fit(s_filename_part).transform(s_filename_part)
train_data['filename_part5'] = s_filename_part

s_filename_part = train_data['filename_part6']
s_filename_part = LabelEncoder().fit(s_filename_part).transform(s_filename_part)
train_data['filename_part6'] = s_filename_part

s_filename_part = train_data['filename_part7']
s_filename_part = LabelEncoder().fit(s_filename_part).transform(s_filename_part)
train_data['filename_part7'] = s_filename_part

s_filename_part = train_data['filename_part8']
s_filename_part = LabelEncoder().fit(s_filename_part).transform(s_filename_part)
train_data['filename_part8'] = s_filename_part

s_filename_part = train_data['filename_part9']
s_filename_part = LabelEncoder().fit(s_filename_part).transform(s_filename_part)
train_data['filename_part9'] = s_filename_part

s_label = train_data['BatchFile']
s_label = LabelEncoder().fit(s_label).transform(s_label)
train_data['BatchFile'] = s_label

x = train_data.values[1::, 0:12]
y = train_data.values[1::, 12].astype(np.int)

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.5, random_state=0)

classfier = RandomForestClassifier(n_estimators=100)
classfier.fit(x_train, y_train)
scores = classfier.score(x_test, y_test)
print scores








