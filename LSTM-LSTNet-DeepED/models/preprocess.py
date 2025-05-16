import numpy as np
from tqdm import tqdm
import tensorflow as tf

from configs.constant_glob import *

def _float_feature(value):
  return tf.train.Feature(
    float_list=tf.train.FloatList(value=value.reshape(-1))
  )

def feature_example(x, y):
  feature = {
    'y': _float_feature(y),
    'x': _float_feature(x),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))

def convert2TFR(np_x, np_y, output_path):
  assert np_x.shape[0] == np_y.shape[0]
  
  num = np_x.shape[0]
  with tf.io.TFRecordWriter(output_path) as all_writer:
    for i in tqdm(range(num), desc='Writing to TFRecords ...'): 
      tf_example = feature_example(np_x[i], np_y[i])
      all_writer.write(tf_example.SerializeToString())
            
            
