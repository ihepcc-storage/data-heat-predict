import numpy as np

import tempfile
import sys
import time
import numpy
import argparse
from tensorflow.contrib.learn.python.learn\
    import metric_spec
#from tensorflow.contrib.learn.python.learn.estimators\
    #import estimator
from tensorflow.contrib.tensor_forest.client\
    import eval_metrics
from tensorflow.contrib.tensor_forest.client\
    import random_forest
from tensorflow.contrib.tensor_forest.python\
    import tensor_forest
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.platform import app

import BuildDataSet

'''
import affinity
import multiprocessing
affinity.set_process_affinity_mask(0,2**multiprocessing.cpu_count()-1)
'''

FLAGS = None

## Random Forest Classifier
def build_extimator(num_classes, num_features, model_dir):
    """Build an estimator"""
    params = tensor_forest.ForestHParams(
        num_classes=num_classes,
        num_features=num_features,
        num_trees=FLAGS.num_trees,
        max_nodes=FLAGS.max_nodes)
    graph_builder_class = tensor_forest.RandomForestGraphs
    if FLAGS.use_training_loss:
        print 'use training loss'
        time.sleep(5)
        graph_builder_class = tensor_forest.TrainingLossForest
    # Use the SKCompat wrapper, which gives us a convenient way to split
    # in-memory data like MINIST into batches.
    return random_forest.TensorForestEstimator(
        params, graph_builder_class=graph_builder_class,
        model_dir=model_dir,num_trainers=1
    )


def train_and_eval():
    model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
    print "model_directory = %s" % model_dir



    EosDataSet = BuildDataSet.read_data_sets(FLAGS.train_time_begin,FLAGS.train_time_end,
                                              FLAGS.test_time_begin,FLAGS.test_time_end)

    train_input_fn = numpy_io.numpy_input_fn(
        x = {'features':EosDataSet[0].astype(numpy.float32)},
        #x = EosDataSet[0].astype(numpy.float64),
        y = EosDataSet[1].astype(numpy.int32),
        batch_size=FLAGS.batch_size,
        num_epochs=None,
        shuffle=True)

    print "Feature example:", EosDataSet[0][0]
    est = build_extimator(3,len(EosDataSet[0][0]),model_dir)


    """
    #"test"
    EosDataSetTest =FeatureProcessFromFilename.read_data_sets(1511107200,1511193600,['/eos/user/x/xuhs/booster_TMCI_20170726_lat_xiy_0_Nonlinear_1p5mm/Results_15f0_16385bins_1000000MP/18.00nC/wp00_s.txt',
                                          '/eos/user/x/xuhs/booster_TMCI_20170726_lat_xiy_0_Nonlinear_1p5mm/Results_15f0_16385bins_1000000MP/18.00nC/mwitrack.twi'])
    x = {'features':EosDataSetTest[0].astype(numpy.float32)}
    est = build_extimator(3, len(EosDataSetTest[0][0]), model_dir)
    predictions = est.predict(x=x)
    predict_result = list(predictions)
    for r in predict_result:
        print r
    return 0
    #"test end."
    """

    est.fit(input_fn=train_input_fn, steps=None)


    #est.fit(x=EosDataSet[0].astype(numpy.float32),y=EosDataSet[1],batch_size=FLAGS.batch_size)

    metric_name = 'accuracy'
    metric = {
        metric_name:
              metric_spec.MetricSpec(
                    eval_metrics.get_metric(metric_name),
                    prediction_key=eval_metrics.get_prediction_key(metric_name))
    }

    test_input_fn = numpy_io.numpy_input_fn(
        x = {'features':EosDataSet[2].astype(numpy.float32)},
        y = EosDataSet[3].astype(numpy.int32),
        batch_size=FLAGS.batch_size,
        num_epochs=1,
        shuffle=False)

    #results = est.score(x=EosDataSet[2], y=EosDataSet[3],
    #                    batch_size=FLAGS.batch_size,
    #                    metrics=metric)
    results = est.evaluate(input_fn=test_input_fn, metrics=metric)

    for key in sorted(results):
        print '%s : %s' % (key,results[key])
    return est


def test(est = None):
    """Test from the input."""



def main():
    train_and_eval()
    test()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./Model',
        help='Base directory for output models.'
    )
    parser.add_argument(
        '--train_steps',
        type=int,
        default=1000,
        help='Number of training steps.'
    )
    parser.add_argument(
        '--batch_size',
        type=str,
        default=1000,
        help='Number of examples in a training batch.'
    )
    parser.add_argument(
        '--num_trees',
        type=int,
        default=100,
        help="Number of trees in the forest."
    )
    parser.add_argument(
        '--max_nodes',
        type=int,
        default=1000,
        help='Max total nodes in a single tree.'
    )
    parser.add_argument(
        '--use_training_loss',
        type=bool,
        default=True,
        help='If ture, use training loss as terminaiton ctiteria.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.model_dir = './Model'
    FLAGS.batch_size = 10
    FLAGS.num_trees = 1000
    FLAGS.max_nodes = 1000
    FLAGS.use_training_loss = True

    FLAGS.train_time_end = 9999999999.99-1519833600.00
    FLAGS.train_time_begin = 9999999999.99-1560179200.00
    FLAGS.test_time_end = 9999999999.99-1519833600.00
    FLAGS.test_time_begin = 9999999999.99-1560179200.00

    #app.run(main=main,argv=[sys.argv[0] + unparsed])
    main()
