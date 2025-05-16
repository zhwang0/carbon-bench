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
            
            
def freezeModel(ls_model, except_idx):
  print('======== Start finetuning ============')
  print('Freezing all layers except ', TREE_BAND[except_idx-1])
  for i, model in enumerate(ls_model): 
    if i != except_idx: 
      model.trainable = False
      
      for layer in model.layers:
        layer.trainable = False
        
        
def freezeModelClass(model, ls_unfreeze_idx):
  print('======== Start finetuning ============')
  print('Freezing all layers except ')
  for i in ls_unfreeze_idx:
    print(TREE_BAND[i])
  
  model.m_ecos.trainable = False
  num_out = len(model.ls_dense_final)
  
  for i in range(num_out): 
    if i not in ls_unfreeze_idx:
      model.ls_dense1[i].trainable = False
      model.ls_dense2[i].trainable = False
      model.ls_dense_final[i].trainable = False
      
      for j in range(len(model.ls_branch_lstms[i])):
        model.ls_branch_lstms[i][j].trainable = False
        model.ls_branch_lstms[i][j].trainable = False
        model.ls_branch_lstms[i][j].trainable = False

def unfreezeModelClass(model):
  print('Unfreezing all layers.')
  for layer in model.layers:
    layer.trainable = True
    # Check if the layer itself has other layers 
    # (e.g., Sequential or functional API nested models)
    if hasattr(layer, 'layers'):
        unfreezeModelClass(layer)
            




  
            
