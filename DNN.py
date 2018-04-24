from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib import learn

import os
import urllib


import numpy as np


# Specify that all features have real-value data
feature_columns = [layers.real_valued_column("",dimension=9)]

# Build 3 layer DNN with 10,20,10 units respectively.
classifier = learn.DNNClassifier(feature_columns=feature_columns,
                                 hidden_units=[10,20,10],
                                 n_classes=2,
                                 model_dir="./DNN_Model")
