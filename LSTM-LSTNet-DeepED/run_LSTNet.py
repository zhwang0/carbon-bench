#!/usr/bin/env python
# coding: utf-8

# # DeepED Global
# 



# In[16]:


import numpy as np
import pandas as pd
import time
import os
import gc
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
from glob import glob
import argparse

from configs.constant_glob import *
from utils.func import *
from utils.preprocess import generateTrainTestGrid
from models.process import *
from models.model import *
# from models.model_finetune import *


# In[17]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

assignGPU(0)


# In[18]:


code_version = 'deeped_v4_age10'

# notebook
model_name = 'M220'
model_idx = 0

age_start_idx = 1
age_end_idx = 2
year_len = 1

BATCH_SIZE = 60 # 4 in pc, 20 in server
n_epoch = 60
initial_epoch = 0
finetune_start_epoch = 0 # 0 means no finetune

initial_learning_rate = 0.0001
finetune_learning_rate = 0.00001
decay_steps = 1000 # cahnge it when change batch size and dataset
max_to_keep = 50
w_cumul = 0.6 # 0.75 seems best
w_delta = 0. 

n_lstm_eco = 1
n_lstm_height = 1

finetune_target_index = [0]
finetune_2step = 0 # 1 means use 2-step finetune, 0 means not use
pass_state = 0 # 1 means pass states in LSTM, 0 means not pass
train_pred = 0 # 1 means train prediction, 0 means not train


path_pretrain = os.path.join('pretrain','V2M65','ckpt-26')
path_pretrain = os.path.join('pretrain','V3M101','ckpt-35')
if finetune_start_epoch == 0:
  path_pretrain = ''
path_pretrain = ''
# path_pretrain = os.path.join('pretrain','M209_dup2','ckpt-11')


agetri_idx = 0 # 1 means use age triplet for each year, 0 means use normal age triplet
if agetri_idx:
  agetri_filename = 'age_triplet_year.npz' 
  
roi_idx = -1 # [0,1,2,3,4,5] means use roi, -1 means not use
if roi_idx == -1:
  train_filename = 'train.tfrecords'
  test_filename = 'test.tfrecords'
  eval_filename = 'res_train4_test8.npz'
  agetri_filename = 'age_triplet.npz'
else: 
  train_filename = f'train_roi1{roi_idx}.tfrecords'
  test_filename = f'test_roi1{roi_idx}.tfrecords'
  eval_filename = f'test_roi1{roi_idx}.npz'
  agetri_filename = 'age_triplet_roi.npz'
stat_filename = 'data_stats.npz'


#-- resutl output path
DIR_INT = os.path.join('datasets', 'global')
DIR_PLOT = 'plots'
DIR_LOG = 'logs'
DIR_OUT = 'results'
if not os.path.exists(DIR_OUT):
  os.makedirs(DIR_OUT)
if not os.path.exists(DIR_PLOT):
  os.makedirs(DIR_PLOT)
if not os.path.exists(DIR_LOG):
  os.makedirs(DIR_LOG)


# In[ ]:


# pyfile
parser = argparse.ArgumentParser(description='Train a model with different parameters.')
parser.add_argument('--mname', type=str, default='default_model', help='Initial learning rate for training')
parser.add_argument('--midx', type=int, default=0, help='Model type to be created')
parser.add_argument('--ias', type=int, default=0, help='Index of age start')
parser.add_argument('--iad', type=int, default=15, help='Index of age end')
parser.add_argument('--n_eco', type=int, default=1, help='Number of LSTM layers of ecos features')
parser.add_argument('--n_hei', type=int, default=1, help='Number of LSTM layers of height branch')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--pass_state', type=int, default=0, help='If pass states in LSTM')
parser.add_argument('--train_pred', type=int, default=0, help='If train with prediction')
parser.add_argument('--ft_tidx', nargs='+', type=int, default=0, help='A index list of target variables to finetune')
parser.add_argument('--ylen', type=int, default=1, help='Length of year in training')
parser.add_argument('--add_noise', type=int, default=0, help='Whether to add noise')
parser.add_argument('--noise_std', type=float, default=1e-4, help='Noise std')
# parser.add_argument('--epoch', type=int, default=100, help='Epoch for training')
# parser.add_argument('--init_epoch', type=int, default=1, help='Initial epoch for training')
# parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for training')
# parser.add_argument('--maxkeep', type=int, default=20, help='Maximum model to keep')
# parser.add_argument('--wc', type=float, default=0., help='Weights of cumulative loss')
# parser.add_argument('--wd', type=float, default=0., help='Weights of delta loss')
# parser.add_argument('--pretrain', type=str, default='', help='Path to pretrained model')
# parser.add_argument('--ft_epoch', type=int, default=0, help='The epoch to start finetune')
# parser.add_argument('--ft_lr', type=float, default=0.00001, help='Learning rate for finetune')
# parser.add_argument('--ft_2step', type=int, default=1, help='If use 2-step finetune')
# parser.add_argument('--roi_idx', type=int, default=-1, help='Which roi to use, -1 means not use')
# parser.add_argument('--agetri_idx', type=int, default=0, help='Which age triplet to use, 1 means using yearly age triplet, 0 means normal')
args = parser.parse_args()

model_name = args.mname
model_idx = args.midx
year_len = args.ylen
age_start_idx = args.ias 
age_end_idx = args.iad
n_lstm_eco = args.n_eco
n_lstm_height = args.n_hei
BATCH_SIZE = args.batch_size
pass_state = args.pass_state
finetune_target_index = args.ft_tidx
train_pred = args.train_pred
add_noise = args.add_noise
noise_std = args.noise_std
# n_epoch = args.epoch
# initial_epoch = args.init_epoch
# initial_learning_rate = args.lr
# max_to_keep = args.maxkeep 
# w_cumul = args.wc 
# w_delta = args.wd 
# path_pretrain = args.pretrain 
# finetune_start_epoch = args.ft_epoch 
# finetune_learning_rate = args.ft_lr
# finetune_2step = arg.ft_2step
# roi_idx = arg.roi_idx
# agetri_idx = arg.agetri_idx


# In[19]:


#-- set up logging
logging.basicConfig(filename=os.path.join(DIR_LOG, code_version+'_'+model_name+'.log'), level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

logging.info('\n==========Model Details=============')
logging.info(f'=== Code version: {code_version}')
logging.info(f'=== Model name: {model_name}')
logging.info(f'=== Model type index: {model_idx}')
logging.info(f'=== # LSTM layers for ecos: {n_lstm_eco}')
logging.info(f'=== # LSTM layers for height: {n_lstm_height}')
logging.info('\n')
logging.info(f'=== Total epoch: {n_epoch}')
logging.info(f'=== Initial epoch: {initial_epoch}')
logging.info(f'=== Initial learning rate: {initial_learning_rate}')
logging.info(f'=== Batch size: {BATCH_SIZE}')
logging.info(f'=== Max model to keep: {max_to_keep}')
logging.info(f'=== Weights of cumulative loss: {w_cumul}')
logging.info(f'=== Weights of delta loss: {w_delta}')
logging.info('\n')
logging.info(f'=== If pass states in LSTM: {pass_state}')
logging.info(f'=== If train with prediction: {train_pred}')
logging.info(f'=== Load pretrain checkpoint from: {path_pretrain}')
logging.info(f'=== Target variable to finetune: {[TREE_BAND[ele] for ele in finetune_target_index]}')
logging.info(f'=== Epoch starting finetuning: {finetune_start_epoch}')
logging.info(f'=== Finetune learning rate: {finetune_learning_rate}')
logging.info(f'=== If use 2-step finetune: {finetune_2step}')
logging.info('\n')
logging.info(f'=== Age starts from: {age_start_idx}')
logging.info(f'=== Age ends at: {age_end_idx}')
logging.info('\n')
logging.info(f'=== ROI region index: {roi_idx}')
logging.info(f'=== Train filename: {train_filename}')
logging.info(f'=== Validation filename: {test_filename}')
logging.info(f'=== Evaluation filename: {eval_filename}')
logging.info(f'=== Data stats filename: {stat_filename}')
logging.info(f'=== Age triplet filename: {agetri_filename}')
logging.info('====================================\n')


# # Preprocess

# In[20]:


def input_pipeline(filename, batch_size, is_shuffle=True, is_stdz=True, is_dup=True, age_start_idx=0, age_end_idx=15):
  feature_description = {
    'y': tf.io.VarLenFeature(tf.float32),
    'x': tf.io.VarLenFeature(tf.float32),
  }
      
  def _parse_function(example_proto):
    feature_dict = tf.io.parse_single_example(example_proto, feature_description)

    x = tf.sparse.to_dense(feature_dict['x'], default_value=0)
    x = tf.reshape(x, [N_YEAR, N_T, N_CONS])

    y = tf.sparse.to_dense(feature_dict['y'], default_value=0)
    y = tf.reshape(y, [N_AGE, N_YEAR+1, N_T, N_OUT])
    y = tf.transpose(y, (1,0,2,3))
    
    if is_stdz: 
      x = normData(x, X_MEAN, X_STD)
      y = normData(y, Y_MEAN, Y_STD)
      
    if is_dup: 
      # duplicate Y for annual prediction
      y = dupMonth_TF(y, dup_idx=11)

    # one-step-ahead prediciton
    x = addInitY2X_TF(x, y)
    y = y[1:]

    # add age
    x = addAgeTriplet_TF(x, TRI_AGE,
                         normData(TRI_MEAN, Y_MEAN, Y_STD),
                         normData(TRI_SLOPE, Y_MEAN, Y_STD))

    
    x = tf.transpose(x, (1,0,2,3))
    y = tf.transpose(y, (1,0,2,3)) # [N_AGE, N_YEAR, N_T, N_OUT]
    
    x = x[age_start_idx:age_end_idx]
    y = y[age_start_idx:age_end_idx]
        
    x = tf.reshape(x, (-1,N_T,N_FEA))
    y = tf.reshape(y, (-1,N_T,N_OUT))
    
    y = y[:,-1] # only use the last month dimension

    return x, y

  dataset = tf.data.TFRecordDataset(filename)
  if is_shuffle: 
    dataset = dataset.shuffle(buffer_size=2000)
  batch = (
    dataset
    .map(_parse_function)
    .batch(batch_size, drop_remainder=True)
    .prefetch(buffer_size=tf.data.AUTOTUNE))  
  
  return batch


# In[21]:


isDup = True

# read stat data
data_stat = np.load(os.path.join(DIR_INT, stat_filename))
X_MEAN = data_stat['x_mean']
Y_MEAN = data_stat['y_mean']
X_STD = data_stat['x_std']
Y_STD = data_stat['y_std']

# read age
age_triplet = np.load(os.path.join(DIR_INT, agetri_filename))
TRI_AGE = age_triplet['age']
TRI_MEAN = age_triplet['age_mean']
TRI_SLOPE = age_triplet['age_slope']
# select roi
if roi_idx != -1:
  TRI_MEAN = TRI_MEAN[:,roi_idx]
  TRI_SLOPE = TRI_SLOPE[:,roi_idx]
  

# generate tf_records if not exists
if not os.path.exists(os.path.join(DIR_INT, train_filename)):   
  data_train = np.load(os.path.join(DIR_INT, 'res_train4_test8.npz'))

  convert2TFR(data_train['x_test'], 
              np.transpose(data_train['y_test'], (1,0,2,3,4)), 
              os.path.join(DIR_INT, 'test.tfrecords'))
  convert2TFR(data_train['x_train'], 
              np.transpose(data_train['y_train'], (1,0,2,3,4)), 
              os.path.join(DIR_INT, 'train.tfrecords'))

else: 
  train_batch = input_pipeline(os.path.join(DIR_INT, train_filename), BATCH_SIZE, age_start_idx=age_start_idx, age_end_idx=age_end_idx)
  vali_batch = input_pipeline(os.path.join(DIR_INT, test_filename), BATCH_SIZE, age_start_idx=age_start_idx, age_end_idx=age_end_idx)
  

# select age
TRI_AGE = TRI_AGE[age_start_idx:age_end_idx]
TRI_MEAN = TRI_MEAN[age_start_idx:age_end_idx]
TRI_SLOPE = TRI_SLOPE[age_start_idx:age_end_idx]
N_AGE = TRI_AGE.size
print(f'Working on age: {TRI_AGE}')
logging.info(f'Working on age: {TRI_AGE}')

age_tri = [TRI_AGE, TRI_MEAN, TRI_SLOPE]
data_stat = [X_MEAN, X_STD, Y_MEAN, Y_STD]
  


# # Train

# In[22]:


def computeMSE(y_true, y_pred):
  return tf.math.reduce_mean((y_true-y_pred)**2)

def computeCumLoss(y_true, y_pred):     
  diff = y_true - y_pred
  diff = tf.reshape(diff, [-1, N_YEAR, N_OUT])
  diff = tf.math.reduce_sum(diff, 1)

  return tf.math.reduce_mean(diff**2)

def computeCumLossNYears(y_true, y_pred, num_year):
  # num_year < N_YEAR
  diff = y_true - y_pred
  diff = tf.reshape(diff, [-1, N_YEAR, N_OUT])

  reminder_idx = int(N_YEAR/num_year)*num_year
  diff1 = tf.reshape(diff[:,:reminder_idx], [-1, num_year, N_OUT])
  diff2 = diff[:,reminder_idx:]
  diff1 = tf.math.reduce_sum(diff1, 1)
  diff2 = tf.math.reduce_sum(diff2, 1)

  return tf.math.reduce_mean(diff1**2)+tf.math.reduce_mean(diff2**2)

def computeCumLossNYearsOverlap(y_true, y_pred):
  '''
  num_year_segment: a year window for sliding entire time series
  we set a fixed number here in order to avoid using for loop in a tensorflow function
  '''
  
  diff = y_true - y_pred
  diff = tf.reshape(diff, [BATCH_SIZE, -1, N_OUT])
  
  num_year = diff.shape[1]
  num_year_segment = 5
  num_pairs = num_year - num_year_segment + 1
  diff = tf.repeat(diff[:,tf.newaxis], num_year_segment, 1)
  d0 = diff[:,0:1,0:num_pairs]
  d1 = diff[:,1:2,1:(num_pairs+1)]
  d2 = diff[:,2:3,2:(num_pairs+2)]
  d3 = diff[:,3:4,3:(num_pairs+3)]
  d4 = diff[:,4:5,4:(num_pairs+4)]
  d = tf.concat([d0, d1], 1)
  d = tf.concat([d, d2], 1)
  d = tf.concat([d, d3], 1)
  d = tf.concat([d, d4], 1)
  
  # sum no-signed loss for all year segments
  d = tf.math.reduce_sum(d, 1)
  
  return tf.math.reduce_mean(d**2)

def computeSingleTargetDeltaLoss(y_true, y_pred, target_idx):    
  y_true = tf.reshape(y_true, [-1, N_YEAR, N_OUT])
  y_pred = tf.reshape(y_pred, [-1, N_YEAR, N_OUT])
  
  # only constraint on agb
  y_true = y_true[...,target_idx]
  y_pred = y_pred[...,target_idx]

  delta_true = y_true[...,1:] - y_true[...,:-1] 
  delta_pred = y_pred[...,1:] - y_pred[...,:-1]
  
  return tf.math.reduce_mean((delta_true-delta_pred)**2)

def computeTotLoss(Y, y_, w_cumul, w_delta): 
  loss_mse = computeMSE(Y, y_)
  
  # return loss_mse, [0, 0, 0]
  
  #-- comparison methods for cumuloss
  # loss_cum = computeCumLoss(Y, y_)
  # loss_cum = computeCumLossNYears (Y, y_, 5)
  
  # loss_del = computeSingleTargetDeltaLoss(Y, y_, 0) # only constraint on height
  # loss_del *= w_delta

  loss_cum = computeCumLossNYearsOverlap(Y, y_)
  loss_cum *= w_cumul
  
  loss_del = 0
  # loss_cum = 0
  tot_loss = loss_mse + loss_cum + loss_del
  
  return tot_loss, [loss_mse, loss_cum, loss_del]


def train_step(MODEL, MODEL_OPZ, X, Y, y_weight):
  with tf.GradientTape() as tape:
    y_ = MODEL(X, training=True)
    tot_loss, sub_loss = computeTotLoss(Y*y_weight, y_*y_weight, w_cumul, w_delta)
  grads = tape.gradient(tot_loss, MODEL.trainable_variables)
  MODEL_OPZ.apply_gradients(zip(grads, MODEL.trainable_variables))
  
  return tot_loss, sub_loss  
 
def val_step(MODEL, X, Y, y_weight):
  y_ = MODEL(X, training=False)
  tot_loss, sub_loss = computeTotLoss(Y*y_weight, y_*y_weight, w_cumul, w_delta)
  return tot_loss, sub_loss


def trainBatch(MODEL, MODEL_OPZ, ls_ckpmg, train_batch, val_batch, total_epoch, initial_epoch, y_weight, year_len=1, add_noise=False, noise_std=1e-4):
  '''
  num_year_segment has to be a factor of N_YEAR
  '''
  loss_train = []
  loss_val = []
  loss_t = []
  min_val = 1e5
    
  for epoch in range(initial_epoch-1, total_epoch):
    start = time.time()
    
    # compute loss for each bactch
    sub_loss = []
    sub_detail = []
    for X, Y in train_batch:
      # X = (BATCH_SIZE, 40, 12, 158), Y = (BATCH_SIZE, 40, 7)
      
      X = tf.reshape(X, (-1,int(N_YEAR/year_len),N_T*year_len,N_FEA))
      Y = Y[:,(year_len-1)::year_len]
      
      if add_noise:
        X = add_random_walk_noise_to_batch(X, noise_std)
        
      X = tf.reshape(X, (-1,N_T*year_len,N_FEA))
      Y = tf.reshape(Y, (-1,N_OUT))
      
      cur_loss, sub_detail_loss = train_step(MODEL, MODEL_OPZ, X, Y, y_weight)
      sub_loss.append(float(cur_loss))
      sub_detail.append(sub_detail_loss)
    train_loss = np.mean(np.array(sub_loss))
    train_detail = np.mean(np.array(sub_detail), axis=0)
    

    #-- val
    # compute loss for each batch
    if val_batch is not None:
      sub_loss = []
      for X, Y in val_batch:
        
        X = tf.reshape(X, (-1,int(N_YEAR/year_len),N_T*year_len,N_FEA))
        Y = Y[:,(year_len-1)::year_len]
        X = tf.reshape(X, (-1,N_T*year_len,N_FEA))
        Y = tf.reshape(Y, (-1,N_OUT))
        
        cur_loss, _ = val_step(MODEL, X, Y, y_weight)
        sub_loss.append(float(cur_loss))
      val_loss = np.mean(np.array(sub_loss))
    else: 
      val_loss = np.nan
        

    #-- save & output
    loss_train.append([train_loss])
    loss_val.append([val_loss])
    loss_t.append([(time.time()-start)/60])
    
    # logging.info training progess every n epochs
    # if epoch % 1 == 0:
    logging.info('Time for epoch {:} is {:.1f} min: train loss = {:.5f} *e^-3, val loss = {:.5f} *e^-3'.format(
        epoch+1, loss_t[-1][0], train_loss*1000, val_loss*1000))
    logging.info('\t: mse loss = {:.5f} *e^-3, cumu loss = {:.5f} *e^-3, delta loss = {:.5f} *e^-3'.format(
        train_detail[0]*1000, train_detail[1]*1000, train_detail[2]*1000))

    # Save best model
    if min_val > val_loss: 
      min_val = val_loss
      for cur_ckpmg in ls_ckpmg: 
          cur_ckpmg.save()
      
      
  # save train loss
  hist1 = pd.DataFrame(np.array(loss_train), columns=['train_loss'])
  hist2 = pd.DataFrame(np.array(loss_val), columns=['val_loss'])
  hist0 = pd.DataFrame(np.array(loss_t), columns=['time_m'])
  train_hist = pd.concat([hist0, hist1], axis=1)
  train_hist = pd.concat([train_hist, hist2], axis=1)

  return train_hist


###====== pass model state in training ======###
def call_step_state(MODEL, X, training=True):
  ls_y_ = []
  for i in range(N_YEAR):
    if i != 0: 
      y_, h_state, c_state = MODEL(X[i], h_state, c_state, training=training)
    else: 
      y_, h_state, c_state = MODEL(X[i], training=training)
    ls_y_.append(y_)      
  return tf.convert_to_tensor(ls_y_)

def train_step_state(MODEL, MODEL_OPZ, X, Y):
  with tf.GradientTape() as tape:
    y_ = call_step_state(MODEL, X, training=True)
    tot_loss, sub_loss = computeTotLoss(Y, y_, w_cumul, w_delta)
  grads = tape.gradient(tot_loss, MODEL.trainable_variables)
  MODEL_OPZ.apply_gradients(zip(grads, MODEL.trainable_variables))
  
  return tot_loss, sub_loss

def val_step_state(MODEL, X, Y):
  y_ = call_step_state(MODEL, X, training=False)
  tot_loss, sub_loss = computeTotLoss(Y, y_, w_cumul, w_delta)
  return tot_loss, sub_loss

def trainBatchState(MODEL, MODEL_OPZ, ls_ckpmg, train_batch, val_batch, total_epoch, initial_epoch):
  loss_train = []
  loss_val = []
  loss_t = []
  min_val = 1e5
    
  for epoch in range(initial_epoch-1, total_epoch):
    start = time.time()
    
    # compute loss for each bactch
    sub_loss = []
    sub_detail = []
    for X, Y in train_batch:
      # X = (None, 600, 12, 158), Y = (None, 600, 7)
      X = tf.reshape(X, (BATCH_SIZE,N_AGE,N_YEAR,N_T,N_FEA))
      Y = tf.reshape(Y, (BATCH_SIZE,N_AGE,N_YEAR,N_OUT))
      X = tf.transpose(X, (2,0,1,3,4))
      Y = tf.transpose(Y, (2,0,1,3))
      X = tf.reshape(X, (N_YEAR,-1,N_T,N_FEA))
      Y = tf.reshape(Y, (N_YEAR,-1,N_OUT))
      
      cur_loss, sub_detail_loss = train_step_state(MODEL, MODEL_OPZ, X, Y)
      sub_loss.append(float(cur_loss))
      sub_detail.append(sub_detail_loss)
    train_loss = np.mean(np.array(sub_loss))
    train_detail = np.mean(np.array(sub_detail), axis=0)
    

    #-- val
    # compute loss for each batch
    if val_batch is not None:
      sub_loss = []
      for X, Y in val_batch:
        X = tf.reshape(X, (BATCH_SIZE,N_AGE,N_YEAR,N_T,N_FEA))
        Y = tf.reshape(Y, (BATCH_SIZE,N_AGE,N_YEAR,N_OUT))
        X = tf.transpose(X, (2,0,1,3,4))
        Y = tf.transpose(Y, (2,0,1,3))
        X = tf.reshape(X, (N_YEAR,-1,N_T,N_FEA))
        Y = tf.reshape(Y, (N_YEAR,-1,N_OUT))
        
        cur_loss, _ = val_step_state(MODEL, X, Y)
        sub_loss.append(float(cur_loss))
      val_loss = np.mean(np.array(sub_loss))
    else: 
      val_loss = np.nan
        

    #-- save & output
    loss_train.append([train_loss])
    loss_val.append([val_loss])
    loss_t.append([(time.time()-start)/60])
    
    # logging.info training progess every n epochs
    # if epoch % 1 == 0:
    logging.info('Time for epoch {:} is {:.1f} min: train loss = {:.5f} *e^-3, val loss = {:.5f} *e^-3'.format(
        epoch+1, loss_t[-1][0], train_loss*1000, val_loss*1000))
    logging.info('\t: mse loss = {:.5f} *e^-3, cumu loss = {:.5f} *e^-3, delta loss = {:.5f} *e^-3'.format(
        train_detail[0]*1000, train_detail[1]*1000, train_detail[2]*1000))

    # Save best model
    if min_val > val_loss: 
      min_val = val_loss
      for cur_ckpmg in ls_ckpmg: 
          cur_ckpmg.save()
      
      
  # save train loss
  hist1 = pd.DataFrame(np.array(loss_train), columns=['train_loss'])
  hist2 = pd.DataFrame(np.array(loss_val), columns=['val_loss'])
  hist0 = pd.DataFrame(np.array(loss_t), columns=['time_m'])
  train_hist = pd.concat([hist0, hist1], axis=1)
  train_hist = pd.concat([train_hist, hist2], axis=1)

  return train_hist



###====== Use prediction as input for training ======###
def train_step_pred(MODEL, MODEL_OPZ, X, Y, start_idx):
  '''
  start_idx: the index of year, after which use prediction as input, start_idx >= 0
  '''
  with tf.GradientTape() as tape:
    ls_y_ = []
    for i in range(N_YEAR):
      if i>start_idx:
        tmp = tf.repeat(ls_y_[i-1][:,tf.newaxis,:], N_T, axis=-2)
        cur_x = tf.concat([X[i,...,:N_CONS], tmp, X[i,...,N_CONS+N_OUT:]], axis=-1)
      else: 
        cur_x = X[i]
      y_ = MODEL(cur_x, training=True)
      
      ls_y_.append(y_)  
    ls_y_ = tf.convert_to_tensor(ls_y_)
    tot_loss, sub_loss = computeTotLoss(Y, ls_y_, w_cumul, w_delta)
  grads = tape.gradient(tot_loss, MODEL.trainable_variables)
  MODEL_OPZ.apply_gradients(zip(grads, MODEL.trainable_variables))
  
  return tot_loss, sub_loss

def val_step_pred(MODEL, X, Y, start_idx):
  ls_y_ = []
  for i in range(N_YEAR):
    if i>start_idx:
      tmp = tf.repeat(ls_y_[i-1][:,tf.newaxis,:], N_T, axis=-2)
      cur_x = tf.concat([X[i,...,:N_CONS], tmp, X[i,...,N_CONS+N_OUT:]], axis=-1)
    else: 
      cur_x = X[i]
    y_ = MODEL(cur_x, training=True)
    
    ls_y_.append(y_)  
  ls_y_ = tf.convert_to_tensor(ls_y_)
  tot_loss, sub_loss = computeTotLoss(Y, ls_y_, w_cumul, w_delta)
  return tot_loss, sub_loss

def trainBatchPred(MODEL, MODEL_OPZ, ls_ckpmg, train_batch, val_batch, total_epoch, initial_epoch, start_idx):
  loss_train = []
  loss_val = []
  loss_t = []
  min_val = 1e5
    
  for epoch in range(initial_epoch-1, total_epoch):
    start = time.time()
    
    # compute loss for each bactch
    sub_loss = []
    sub_detail = []
    for X, Y in train_batch:
      # X = (None, 600=N_AGE*N_YEAR, 12, 158), Y = (None, 600=N_AGE*N_YEAR, 7)
      X = tf.reshape(X, (BATCH_SIZE,N_AGE,N_YEAR,N_T,N_FEA))
      Y = tf.reshape(Y, (BATCH_SIZE,N_AGE,N_YEAR,N_OUT))
      X = tf.transpose(X, (2,0,1,3,4))
      Y = tf.transpose(Y, (2,0,1,3))
      X = tf.reshape(X, (N_YEAR,-1,N_T,N_FEA))
      Y = tf.reshape(Y, (N_YEAR,-1,N_OUT))

      cur_loss, sub_detail_loss = train_step_pred(MODEL, MODEL_OPZ, X, Y, start_idx)
      sub_loss.append(float(cur_loss))
      sub_detail.append(sub_detail_loss)
    train_loss = np.mean(np.array(sub_loss))
    train_detail = np.mean(np.array(sub_detail), axis=0)
    

    #-- val
    # compute loss for each batch
    if val_batch is not None:
      sub_loss = []
      for X, Y in val_batch:
        X = tf.reshape(X, (BATCH_SIZE,N_AGE,N_YEAR,N_T,N_FEA))
        Y = tf.reshape(Y, (BATCH_SIZE,N_AGE,N_YEAR,N_OUT))
        X = tf.transpose(X, (2,0,1,3,4))
        Y = tf.transpose(Y, (2,0,1,3))
        X = tf.reshape(X, (N_YEAR,-1,N_T,N_FEA))
        Y = tf.reshape(Y, (N_YEAR,-1,N_OUT))
        
        cur_loss, _ = val_step_pred(MODEL, X, Y, start_idx)
        sub_loss.append(float(cur_loss))
      val_loss = np.mean(np.array(sub_loss))
    else: 
      val_loss = np.nan
        

    #-- save & output
    loss_train.append([train_loss])
    loss_val.append([val_loss])
    loss_t.append([(time.time()-start)/60])
    
    # logging.info training progess every n epochs
    # if epoch % 1 == 0:
    logging.info('Time for epoch {:} is {:.1f} min: train loss = {:.5f} *e^-3, val loss = {:.5f} *e^-3'.format(
        epoch+1, loss_t[-1][0], train_loss*1000, val_loss*1000))
    logging.info('\t: mse loss = {:.5f} *e^-3, cumu loss = {:.5f} *e^-3, delta loss = {:.5f} *e^-3'.format(
        train_detail[0]*1000, train_detail[1]*1000, train_detail[2]*1000))

    # Save best model
    if min_val > val_loss: 
      min_val = val_loss
      for cur_ckpmg in ls_ckpmg: 
          cur_ckpmg.save()
      
      
  # save train loss
  hist1 = pd.DataFrame(np.array(loss_train), columns=['train_loss'])
  hist2 = pd.DataFrame(np.array(loss_val), columns=['val_loss'])
  hist0 = pd.DataFrame(np.array(loss_t), columns=['time_m'])
  train_hist = pd.concat([hist0, hist1], axis=1)
  train_hist = pd.concat([train_hist, hist2], axis=1)

  return train_hist



###====== 2-step finetune ======###
# unknown bugs not fixed yet

def computeTotLoss_Branch(Y, y_): 
  loss_mse = computeMSE(Y, y_)
  
  return loss_mse, [0, 0, 0]


def train_step_branch(MODEL, MODEL_OPZ, X, Y):
  with tf.GradientTape() as tape:
    y_ = MODEL(X, training=True)
    tot_loss, sub_loss = computeTotLoss_Branch(Y, y_)
  grads = tape.gradient(tot_loss, MODEL.trainable_variables)
  MODEL_OPZ.apply_gradients(zip(grads, MODEL.trainable_variables))
  
  return tot_loss, sub_loss

def train_step_branch_2step(MODEL, MODEL_OPZ, X, Y):
  with tf.GradientTape() as tape:
    # 1-step
    y_1 = MODEL(X, training=True)
    tot_loss1, _ = computeTotLoss_Branch(Y, y_1)
    
    # 2-step
    X = tf.reshape(X, (-1,N_AGE,N_YEAR,N_T,N_FEA))
    Y = tf.reshape(Y, (-1,N_AGE,N_YEAR,N_OUT))
    y_1 = tf.reshape(y_1, (-1,N_AGE,N_YEAR,N_OUT))
    
    X = X[:,:,1:]
    Y = Y[:,:,1:]
    y_1 = y_1[:,:,:-1]
    
    X1 = X[...,:N_CONS] 
    X2 = X[...,(N_CONS+N_OUT):]
    y_1 = tf.repeat(y_1[...,tf.newaxis,:], N_T, axis=-2)
    X_ = tf.concat([X1, y_1, X2], axis=-1)
    
    X_ = tf.reshape(X_, (-1,N_T,N_FEA))
    Y = tf.reshape(Y, (-1,N_OUT))
    y_2 = MODEL(X_, training=True)
    
    tot_loss2, _ = computeTotLoss_Branch(Y, y_2)
    
    tot_loss = tot_loss1 + tot_loss2
    
  grads = tape.gradient(tot_loss, MODEL.trainable_variables)
  MODEL_OPZ.apply_gradients(zip(grads, MODEL.trainable_variables))
  
  return tot_loss, [tot_loss1, tot_loss2, 0]  
 
def val_step_branch(MODEL, X, Y):
  y_ = MODEL(X, training=False)
  tot_loss, sub_loss = computeTotLoss_Branch(Y, y_)
  return tot_loss, sub_loss

def trainBranch(MODEL, MODEL_OPZ, ls_ckpmg, train_batch, val_batch, total_epoch, initial_epoch):
  loss_train = []
  loss_val = []
  loss_t = []
  min_val = 1e5
    
  for epoch in range(initial_epoch-1, total_epoch):
    start = time.time()
    
    # compute loss for each bactch
    sub_loss = []
    sub_detail = []
    for X, Y in train_batch:
      X = tf.reshape(X, (-1,N_T,N_FEA))
      Y = tf.reshape(Y, (-1,N_OUT))
      
      if finetune_2step: 
        cur_loss, sub_detail_loss = train_step_branch_2step(MODEL, MODEL_OPZ, X, Y)
      else:
        cur_loss, sub_detail_loss = train_step_branch(MODEL, MODEL_OPZ, X, Y)
      sub_loss.append(float(cur_loss))
      sub_detail.append(sub_detail_loss)
    train_loss = np.mean(np.array(sub_loss))
    train_detail = np.mean(np.array(sub_detail), axis=0)
    

    #-- val
    # compute loss for each batch
    if val_batch is not None:
      sub_loss = []
      for X, Y in val_batch:
        X = tf.reshape(X, (-1,N_T,N_FEA))
        Y = tf.reshape(Y, (-1,N_OUT))
        
        cur_loss, _ = val_step_branch(MODEL, X, Y)
        sub_loss.append(float(cur_loss))
      val_loss = np.mean(np.array(sub_loss))
    else: 
      val_loss = np.nan
        

    #-- save & output
    loss_train.append([train_loss])
    loss_val.append([val_loss])
    loss_t.append([(time.time()-start)/60])
    
    # logging.info training progess every n epochs
    # if epoch % 1 == 0:
    logging.info('Time for epoch {:} is {:.1f} min: train loss = {:.5f} *e^-3, val loss = {:.5f} *e^-3'.format(
        epoch+1, loss_t[-1][0], train_loss*1000, val_loss*1000))
    logging.info('\tmse loss = {:.5f} *e^-3, cumu loss = {:.5f} *e^-3, delta loss = {:.5f} *e^-3'.format(
        train_detail[0]*1000, train_detail[1]*1000, train_detail[2]*1000))

    # Save best model
    if min_val > val_loss: 
      min_val = val_loss
      for cur_ckpmg in ls_ckpmg: 
          cur_ckpmg.save()
      
      
  # save train loss
  hist1 = pd.DataFrame(np.array(loss_train), columns=['train_loss'])
  hist2 = pd.DataFrame(np.array(loss_val), columns=['val_loss'])
  hist0 = pd.DataFrame(np.array(loss_t), columns=['time_m'])
  train_hist = pd.concat([hist0, hist1], axis=1)
  train_hist = pd.concat([train_hist, hist2], axis=1)

  return train_hist


# In[14]:


# del MODEL, MODEL_OPTZ, MODEL_CKP, MODEL_CKP_MANAGER


# In[23]:


# initialize
res_output = os.path.join(DIR_OUT, code_version)
if not os.path.exists(res_output): 
    os.makedirs(res_output)
dir_ckpt = os.path.join(res_output, model_name)
print('\nModel output to ', dir_ckpt, '\n')
logging.info(f'\nModel output to {dir_ckpt}\n')


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
  initial_learning_rate,
  decay_steps=decay_steps,
  decay_rate=0.96,
  staircase=True)

# setup weight for each target
# y_weight = tf.convert_to_tensor(Y_MEAN / np.sum(Y_MEAN), dtype=tf.float32)
# y_weight *= 10
y_weight = tf.convert_to_tensor(np.ones(N_OUT), dtype=tf.float32)


# if model_idx == 0: 
#   MODEL = MM_LSTM_Age_v4(n_out=N_OUT, n_cons=N_CONS)
# elif model_idx == 1:
#   MODEL = MM_LSTM_Age_v41(n_out=N_OUT, n_cons=N_CONS)
# elif model_idx == 2:
#   MODEL = MM_LSTM_Age_v42(n_out=N_OUT, n_cons=N_CONS)
# elif model_idx == 3:
#   MODEL = MM_LSTM_Age_v43(n_out=N_OUT, n_cons=N_CONS)
# elif model_idx == 4:
#   MODEL = MM_LSTM_Age_v44(n_out=N_OUT, n_cons=N_CONS)
# elif model_idx == 5:
#   MODEL = MM_LSTM_Age_v45(n_out=N_OUT, n_cons=N_CONS)
# elif model_idx == 6:
#   MODEL = MM_LSTM_Age_v46(n_out=N_OUT, n_cons=N_CONS)
# elif model_idx == 7:
#   MODEL = MM_LSTM_Age_v47(n_out=N_OUT, n_cons=N_CONS)
# elif model_idx == 8:
#   MODEL = MM_LSTM_Age_v48(n_out=N_OUT, n_cons=N_CONS)
# else: 
#   raise ValueError('Model type index not found')

MODEL = DeepEDv2_LSTNet_2d(n_out=N_OUT, n_cons=N_CONS)
# MODEL = MM_LSTM_Age_v3(n_out=N_OUT, n_cons=N_CONS)
# MODEL = MM_LSTM_Age_v4(n_out=N_OUT, n_cons=N_CONS)

# MODEL = MM_LSTM_Age_v5(n_out=N_OUT, n_cons=N_CONS)

  
MODEL_OPTZ = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
MODEL_CKP = tf.train.Checkpoint(optimizer=MODEL_OPTZ, model=MODEL)
MODEL_CKP_MANAGER = tf.train.CheckpointManager(MODEL_CKP, dir_ckpt, max_to_keep=max_to_keep) # max_to_keep=None -> to keep all

if not os.path.exists(os.path.join(dir_ckpt, 'loss.csv')): 
  pd_loss = pd.DataFrame(columns=['time_m', 'train_loss', 'val_loss'])
else: 
  pd_loss = pd.read_csv(os.path.join(dir_ckpt, 'loss.csv'))


# restore pretrain weights
if len(path_pretrain)>0: 
  MODEL_CKP.restore(path_pretrain)
  print(f'\nRestore pretrain weights from \n\t{path_pretrain}\n')
  logging.info(f'\nRestore pretrain weights from \n\t{path_pretrain}\n')
  


# In[ ]:


# for prediction in training only
if finetune_start_epoch>0:
  if initial_epoch<finetune_start_epoch:
    
    # --- normal training ---
    if pass_state:
      hist_train = trainBatchState(MODEL, MODEL_OPTZ, [MODEL_CKP_MANAGER], 
        train_batch, vali_batch, finetune_start_epoch, initial_epoch)
    else: 
      hist_train = trainBatch(MODEL, MODEL_OPTZ, [MODEL_CKP_MANAGER], 
        train_batch, vali_batch, finetune_start_epoch, initial_epoch, y_weight, year_len=year_len, 
        add_noise=add_noise, noise_std=noise_std)
    pd_loss = pd.concat([pd_loss, hist_train], axis=0, ignore_index=True)
    
    _ = eval(MODEL, os.path.join(DIR_INT,eval_filename), os.path.join(dir_ckpt, 'after_normal_train'), 
      age_tri, data_stat, [age_start_idx, age_end_idx], 
      isPred2Input=True, isState=pass_state, isTrain=False, isOutput=True)
    
    
    # --- training with prediction ---
    if train_pred: 
      # num_pred_ft_epoch = finetune_start_epoch
      num_pred_ft_epoch = 30
      ls_start_idx = [30,20,10,0]
      for cur_idx, cur_start_idx in enumerate(ls_start_idx):
        hist_train = trainBatchPred(MODEL, MODEL_OPTZ, [MODEL_CKP_MANAGER], 
          train_batch, vali_batch, 
          finetune_start_epoch+num_pred_ft_epoch*(cur_idx+1), 
          finetune_start_epoch+num_pred_ft_epoch*cur_idx, start_idx=cur_start_idx)
        pd_loss = pd.concat([pd_loss, hist_train], axis=0, ignore_index=True)
        
        _ = eval(MODEL, os.path.join(DIR_INT,eval_filename), 
          os.path.join(dir_ckpt, 'after_train_pred'+str(cur_start_idx).zfill(2)), 
          age_tri, data_stat, [age_start_idx, age_end_idx], 
          isPred2Input=True, isState=pass_state, isTrain=False, isOutput=True, y_lim=[0,1.5])
      
      end_epoch = finetune_start_epoch+num_pred_ft_epoch*len(ls_start_idx)
    else: 
      end_epoch = finetune_start_epoch
    
    # --- fine tune ---
    num_ft_epoch = n_epoch - finetune_start_epoch
    for cur_idx, cur_ft_idx in enumerate(finetune_target_index): 
      
      # retrieve the best model and evaluate 
      best_ckp = retrieveBestCKPT(glob(os.path.join(dir_ckpt, 'ckpt-*.index')))
      logging.info(f'\nRestore the best model from \n\t{best_ckp}\n')
      MODEL_CKP.restore(best_ckp)
      
      _ = eval(MODEL, os.path.join(DIR_INT,eval_filename), os.path.join(dir_ckpt, Path(best_ckp).stem), 
        age_tri, data_stat, [age_start_idx, age_end_idx], 
        isPred2Input=True, isState=pass_state, isTrain=False, isOutput=True)
      
      # fine-tuning
      freezeModelClass(MODEL, ls_unfreeze_idx=[cur_ft_idx])
      
      MODEL_OPTZ.learning_rate = finetune_learning_rate
      if pass_state:
        hist_train = trainBatchState(MODEL, MODEL_OPTZ, [MODEL_CKP_MANAGER], 
          train_batch, vali_batch, 
          end_epoch+num_ft_epoch*(cur_idx+1), 
          end_epoch+num_ft_epoch*cur_idx)
      else: 
        hist_train = trainBranch(MODEL, MODEL_OPTZ, [MODEL_CKP_MANAGER], 
          train_batch, vali_batch, 
          end_epoch+num_ft_epoch*(cur_idx+1), 
          end_epoch+num_ft_epoch*cur_idx)
      pd_loss = pd.concat([pd_loss, hist_train], axis=0, ignore_index=True)
    
  else: 
    # --- fine-tuning dirctly ---
    if len(path_pretrain) == 0:
      # restore from current best ckpt 
      best_ckp = retrieveBestCKPT(glob(os.path.join(dir_ckpt, 'ckpt-*.index')))
    else: 
      best_ckp = path_pretrain
    logging.info(f'\nRestore the best model from \n\t{best_ckp}\n')
    MODEL_CKP.restore(best_ckp)
    
    
    num_ft_epoch = n_epoch - finetune_start_epoch
    for cur_idx, cur_ft_idx in enumerate(finetune_target_index): 
      
      freezeModelClass(MODEL, ls_unfreeze_idx=[cur_ft_idx])
      
      MODEL_OPTZ.learning_rate = finetune_learning_rate
      if pass_state: 
        hist_train = trainBatchState(MODEL, MODEL_OPTZ, [MODEL_CKP_MANAGER],          
          train_batch, vali_batch,
          finetune_start_epoch+num_ft_epoch*(cur_idx+1), 
          finetune_start_epoch+num_ft_epoch*cur_idx)
      else: 
        hist_train = trainBranch(MODEL, MODEL_OPTZ, [MODEL_CKP_MANAGER],     
          train_batch, vali_batch,
          finetune_start_epoch+num_ft_epoch*(cur_idx+1), 
          finetune_start_epoch+num_ft_epoch*cur_idx)
      pd_loss = pd.concat([pd_loss, hist_train], axis=0, ignore_index=True)
      
      # retrieve the best model and evaluate 
      best_ckp = retrieveBestCKPT(glob(os.path.join(dir_ckpt, 'ckpt-*.index')))
      logging.info(f'\nRestore the best model from \n\t{best_ckp}\n')
      MODEL_CKP.restore(best_ckp)
      
      _ = eval(MODEL, os.path.join(DIR_INT,eval_filename), os.path.join(dir_ckpt, Path(best_ckp).stem), 
        age_tri, data_stat, [age_start_idx, age_end_idx], 
        isPred2Input=True, isState=pass_state, isTrain=False, isOutput=True)
  
else:
  # training from stratch withou any finetuning
  if pass_state: 
    hist_train = trainBatchState(MODEL, MODEL_OPTZ, [MODEL_CKP_MANAGER],          
      train_batch, vali_batch, n_epoch, initial_epoch)
  else: 
    hist_train = trainBatch(MODEL, MODEL_OPTZ, [MODEL_CKP_MANAGER], 
        train_batch, vali_batch, n_epoch, initial_epoch, y_weight, year_len=year_len)

  pd_loss = pd.concat([pd_loss, hist_train], axis=0, ignore_index=True)

pd_loss.to_csv(os.path.join(dir_ckpt,'loss.csv'), index=False)


# # Evaluation

# In[142]:


best_ckp = retrieveBestCKPT(glob(os.path.join(dir_ckpt, 'ckpt-*.index')))

# path_pretrain = os.path.join('pretrain','V3M101','ckpt-35')
# best_ckp = path_pretrain

print(f'\nRestore weights from \n\t{best_ckp}\n')
logging.info(f'\nRestore weights from \n\t{best_ckp}\n')
MODEL_CKP.restore(best_ckp)


# In[ ]:


_ = eval(MODEL, os.path.join(DIR_INT,eval_filename), 
  os.path.join(dir_ckpt, Path(best_ckp).stem), 
  age_tri, data_stat, [age_start_idx, age_end_idx], 
  isPred2Input=True, isState=pass_state, isTrain=False, isOutput=True)

_ = eval(MODEL, os.path.join(DIR_INT,eval_filename), 
  os.path.join(dir_ckpt, Path(best_ckp).stem)+'test_true', 
  age_tri, data_stat, [age_start_idx, age_end_idx], 
  isPred2Input=False, isState=pass_state, isTrain=False, isOutput=True, y_lim=[0,1.5])

_ = eval(MODEL, os.path.join(DIR_INT,eval_filename), 
  os.path.join(dir_ckpt, Path(best_ckp).stem)+'_train_pred', 
  age_tri, data_stat, [age_start_idx, age_end_idx], 
  isPred2Input=True, isState=pass_state, isTrain=True, isOutput=True)

_ = eval(MODEL, os.path.join(DIR_INT,eval_filename), 
  os.path.join(dir_ckpt, Path(best_ckp).stem)+'_train_true', 
  age_tri, data_stat, [age_start_idx, age_end_idx], 
  isPred2Input=False, isState=pass_state, isTrain=True, isOutput=True, y_lim=[0,1.5])


# In[ ]:


# _ = eval_Nyear(MODEL, os.path.join(DIR_INT,eval_filename), 
#   os.path.join(dir_ckpt, Path(best_ckp).stem)+'_'+str(year_len)+'year_test_true', 
#   age_tri, data_stat, [age_start_idx, age_end_idx], num_year_segment=year_len,
#   isPred2Input=False, isState=pass_state, isTrain=False, isOutput=True, y_lim=[0,1.5])

# _ = eval_Nyear(MODEL, os.path.join(DIR_INT,eval_filename), 
#   os.path.join(dir_ckpt, Path(best_ckp).stem)+'_'+str(year_len)+'year_train_true', 
#   age_tri, data_stat, [age_start_idx, age_end_idx], num_year_segment=year_len,
#   isPred2Input=False, isState=pass_state, isTrain=True, isOutput=True, y_lim=[0,1.5])

# _ = eval_Nyear(MODEL, os.path.join(DIR_INT,eval_filename), 
#   os.path.join(dir_ckpt, Path(best_ckp).stem)+'_'+str(year_len)+'year_test_pred', 
#   age_tri, data_stat, [age_start_idx, age_end_idx], num_year_segment=year_len,
#   isPred2Input=True, isState=pass_state, isTrain=False, isOutput=True)

# _ = eval_Nyear(MODEL, os.path.join(DIR_INT,eval_filename), 
#   os.path.join(dir_ckpt, Path(best_ckp).stem)+'_'+str(year_len)+'year_train_pred', 
#   age_tri, data_stat, [age_start_idx, age_end_idx], num_year_segment=year_len,
#   isPred2Input=True, isState=pass_state, isTrain=True, isOutput=True)


# In[ ]:


import xarray as xr



def convertNPY2NC(inp_npy, mask_glob): 
  res_out2d = convertAGE2D(inp_npy, mask_glob)

  raw_nc = xr.open_dataset(os.path.join('datasets', 'global', 'data_eval_global', 'ED_restart-40.nc'))
  res_nc = xr.Dataset({'mean_height':(['time','y','x'], res_out2d[...,0]),
                     'agb':(['time','y','x'], res_out2d[...,1]),
                     'soil_C':(['time','y','x'], res_out2d[...,2]),
                     'LAI':(['time','y','x'], res_out2d[...,3]),
                     'GPP_AVG':(['time','y','x'], res_out2d[...,4]),
                     'NPP_AVG':(['time','y','x'], res_out2d[...,5]),
                     'Rh_AVG':(['time','y','x'], res_out2d[...,6]),
                    },
                    coords={'time': raw_nc.coords['time'].values[23::12], # output year only
                            'y': raw_nc.y,
                            'x': raw_nc.x,
                            'band': raw_nc.band,
                            'spatial_ref': raw_nc.spatial_ref})
  return res_nc


# evaluate global data
mask_glob = np.load(os.path.join(DIR_INT,'Mask_all.npy'))
path_x = os.path.join(DIR_INT, 'data_eval_global', 'glob_X_fea.npy')
ls_path_y = glob(os.path.join(DIR_INT, 'data_eval_global', 'glob_Y*.npy'))

for path_y in ls_path_y: 
  path_xy = [path_x, path_y]
  tmp_out = eval_glob(MODEL, DIR_INT, path_xy, dir_ckpt+'_glob', age_tri, data_stat, isDup=True, isOutput=True)
  convertNPY2NC(tmp_out, mask_glob).to_netcdf(os.path.join(dir_ckpt+'_glob','age'+path_y.split('glob_Y')[1].split('.')[0]+'_pred.nc'))
  


# evaluation
dir_out_eval = os.path.join('results', 'NeurIPS', 'M605_00NO-3_glob') # modify later
dir_out_eval = dir_ckpt+'_glob'

ls_path_true = glob(os.path.join(DIR_INT, 'data_eval_global', 'glob_Y*.npy'))
ls_path_pred = glob(os.path.join(dir_out_eval, '*.npy'))
print(f'Length of true and pred: {len(ls_path_true)}, {len(ls_path_pred)}')

# find test samples 
mask_glob = np.load(os.path.join(DIR_INT,'Mask_all.npy')).astype(int)
mask_train_test = np.load(os.path.join(DIR_INT,'train_test_mask.npy'))
mask_glob[mask_train_test==1] = 2
mask_glob = mask_glob[mask_glob>0]
test_site = mask_glob == 1


import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(true, pred):
  mae = mean_absolute_error(true, pred)
  mse = mean_squared_error(true, pred)
  rmse = np.sqrt(mse)
  
  # Manual R² computation
  ss_total = np.sum((true - np.mean(true))**2)
  ss_residual = np.sum((true - pred)**2)
  r2 = 1 - (ss_residual / ss_total)
  
  return mae, mse, rmse, r2, ss_total, ss_residual

def aggregate_metrics(metrics_list):
  maes, mses, rmses, r2s, ss_totals, ss_residuals = zip(*metrics_list)
  overall_mae = np.mean(maes)
  overall_mse = np.mean(mses)
  overall_rmse = np.mean(rmses)

  # Overall R² computation
  total_ss_total = np.sum(ss_totals)
  total_ss_residual = np.sum(ss_residuals)
  overall_r2 = 1 - (total_ss_residual / total_ss_total)

  return overall_mae, overall_mse, overall_rmse, overall_r2

def compute_metrics_for_each_target(true, pred):
  num_targets = true.shape[-1]
  metrics_by_target = []

  for target in range(num_targets):
    true_target = true[:, :, target].flatten()
    pred_target = pred[:, :, target].flatten()
    metrics = compute_metrics(true_target, pred_target)
    metrics_by_target.append(metrics)

  return metrics_by_target


overall_metrics_by_target = []
individual_metrics_by_age = []

for idx, (path_true, path_pred) in tqdm(enumerate(zip(ls_path_true, ls_path_pred))):
  cur_age = path_true.split('glob_Y')[1].split('.')[0]
  
  true = np.load(path_true)
  pred = np.load(path_pred)
  
  true = true[:,1:,-1]
  pred = np.transpose(pred, (1,0,2))
  true = true[test_site]
  pred = pred[test_site]
  # print(f'True shape: {true.shape}, Pred shape: {pred.shape}')
  
  metrics_by_target = compute_metrics_for_each_target(true, pred)
  overall_metrics_by_target.append(metrics_by_target)
  individual_metrics_by_age.append((cur_age, metrics_by_target))
  
# Aggregate metrics across all pairs for each target
num_targets = len(overall_metrics_by_target[0])
overall_aggregated_metrics = []
for target in range(num_targets):
  target_metrics_list = [pair_metrics[target] for pair_metrics in overall_metrics_by_target]
  aggregated_metrics = aggregate_metrics(target_metrics_list)
  overall_aggregated_metrics.append(aggregated_metrics)

# Prepare the DataFrame for global aggregated metrics
df_global_metrics = pd.DataFrame(
  overall_aggregated_metrics, 
  columns=['MAE', 'MSE', 'RMSE', 'R²'],
  index=TREE_BAND[:num_targets]
)
df_global_metrics.index.name = 'Target'
df_global_metrics['Type'] = 'Global'

# Prepare the DataFrame for individual metrics by age
individual_metrics_data = []
for age, metrics_by_target in individual_metrics_by_age:
  for target, metrics in enumerate(metrics_by_target):
    row = {
      'Age': age,
      'Target': TREE_BAND[target],
      'MAE': metrics[0],
      'MSE': metrics[1],
      'RMSE': metrics[2],
      'R²': metrics[3]
    }
    individual_metrics_data.append(row)

df_individual_metrics = pd.DataFrame(individual_metrics_data)
df_individual_metrics['Type'] = 'Individual'

# Combine the global and individual metrics DataFrames
df_combined = pd.concat([df_global_metrics.reset_index(), df_individual_metrics], ignore_index=True)
df_combined = df_combined[['Type', 'Target', 'Age', 'MAE', 'MSE', 'RMSE', 'R²']]

print(df_combined)
df_combined.to_csv(dir_out_eval+'eval.csv', index=False)

