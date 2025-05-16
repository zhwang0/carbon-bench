import os
import numpy as np
import pandas as pd
import time
import gc
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from configs.constant_glob import *


def generate_random_walk_noise(shape, std):
  random_steps = tf.random.normal(shape, mean=0.0, stddev=std)
  random_walk_noise = tf.cumsum(random_steps, axis=1)
  return random_walk_noise

def add_random_walk_noise_to_batch(batch, std, start_feature=N_CONS, end_feature=N_CONS+N_OUT):
  b, years, months, features = batch.shape

  # Generate random walk noise
  noise_shape = (b, years, end_feature - start_feature)
  random_walk_noise = generate_random_walk_noise(noise_shape, std)

  # replicate and padding
  random_walk_noise = tf.expand_dims(random_walk_noise, axis=2)
  random_walk_noise = tf.tile(random_walk_noise, [1, 1, months, 1])
  padding = [[0, 0], [0, 0], [0, 0], [start_feature, features - end_feature]]
  padded_noise = tf.pad(random_walk_noise, padding)
  
  return batch + padded_noise

def assignGPU(gpu_idx): 
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    print('\n========== GPU Usage =============')
    try:
      tf.config.set_visible_devices(gpus[gpu_idx], 'GPU')
      tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
      visible_gpus = tf.config.get_visible_devices('GPU')
      print(len(gpus), 'physical GPUs in total\nNow using GPU: ', visible_gpus)
    except RuntimeError as e:
      print(e)
    print('==================================\n')


def retrieveBestCKPT(ls_ckp_path):
  tmp = [int(i.split('ckpt-')[-1].split('.')[0]) for i in ls_ckp_path]
  best_ckp = ls_ckp_path[tmp.index(max(tmp))].split('.')[0]
  print('The best model is found at:\n\t', best_ckp)
  return best_ckp


def plotAge(np_rmse, TRI_AGE, isOutput=True, path_out='', y_lim=[0,3.5]):
  # plot age
  num_age = np_rmse.shape[0]
  ls_cor = ['r','g','b','c','y','m','k']
  
  if num_age == 1: 
    fig_r = 1
    fig_c = 2
    figsize = (4,3)
    
    fig = plt.figure(figsize=figsize)
    cur_rmse = np_rmse[0]
    for k in range(N_OUT):
      plt.plot(cur_rmse[:,k], ls_cor[k], label=TREE_BAND[k])
    
    plt.ylim(y_lim)
    plt.title('Age'+str(TRI_AGE[0])) 
    plt.ylabel('RMSE')
    # plt.set_xticks(list(range(0,32,30)))
    # plt.set_xticklabels(list(range(1986,2017,30)))

    plt.legend(loc=0, prop={'size': 7})

  else: 
    fig_r = 3
    fig_c = 5
    figsize = (10,7)
    
    plt.figure()
    fig, axes = plt.subplots(fig_r, fig_c, figsize=figsize)
    for i in range(fig_r):
      for j in range(fig_c):
        if i*fig_c+j < num_age: 
          cur_rmse = np_rmse[i*fig_c+j]
          for k in range(N_OUT):
            axes[i,j].plot(cur_rmse[:,k], ls_cor[k])

          # axes[i,j].set_xticks(list(range(0,32,30)))
          # axes[i,j].set_xticklabels(list(range(1986,2017,30)))
          axes[i,j].set_ylim(y_lim)
          axes[i,j].set_title('Age'+str(TRI_AGE[i*fig_c+j]))
          
      axes[i,0].set_ylabel('RMSE')

    # add legend
    for k in range(N_OUT):
      axes[i,j].plot([0], [0], ls_cor[k], label=TREE_BAND[k])
    axes[i,j].legend(loc=10, prop={'size': 7})   
    
  # save 
  fig.tight_layout()
  if isOutput:
    plt.savefig(path_out+'_fig', bbox_inches='tight')
  print('  Plot done!\n')      

  
def convertRMSE2Table(ls_rmse_all, TRI_AGE): 
  rmse = np.transpose(ls_rmse_all, [0,2,1])
  pd_rmse = pd.DataFrame(rmse.reshape((-1, rmse.shape[2])))
  tmp = pd.DataFrame(np.repeat(TRI_AGE, N_OUT), columns=['age'])
  pd_rmse = pd.concat([tmp, pd_rmse], axis=1)
  return pd_rmse

def convertTable2RMSE(pd_rmse):
  rmse = pd_rmse.iloc[:,1:].values
  rmse = rmse.reshape((N_AGE,N_OUT,N_YEAR))
  rmse = np.transpose(rmse, [0,2,1])
  return rmse


def normData(inp, mean, std): 
  # assert inp.shape[-1] == mean.size
  # assert std.size == mean.size
  
  return (inp-mean)/(std+1e-10)

def invNormData(inp, mean, std): 
  # assert inp.shape[-1] == mean.size
  # assert std.size == mean.size
  
  return inp*(std+1e-10)+mean


def addInitY2X(x, y): 
  out = np.repeat(x[np.newaxis], y.shape[0], axis=0)
  return np.concatenate([out, y[:,:,:-1]], axis=-1)

def addInitY2X_TF(x, y): 
  out = tf.repeat(x[:,tf.newaxis], y.shape[1], axis=1)
  return tf.concat([out, y[:-1]], axis=-1)


def dupMonth(y, dup_idx=11):
  # assert dup_idx < N_T
  
  return np.repeat(y[...,dup_idx:dup_idx+1,:], N_T, axis=-2)

def dupMonth_TF(y, dup_idx=11):
  # assert dup_idx < N_T
  
  return tf.repeat(y[...,dup_idx:dup_idx+1,:], N_T, axis=-2)


def addAgeTriplet(x, np_age, np_mean, np_slope):
  ''' 
  For x with all ages, add age triplet to the last dimension of x
  '''
  # form data structure
  res_mean = np.repeat(np_mean[:,np.newaxis], x.shape[3], axis=1)
  res_mean = np.repeat(res_mean[:,np.newaxis], x.shape[2], axis=1)
  res_mean = np.repeat(res_mean[:,np.newaxis], x.shape[1], axis=1)
  res_slope = np.repeat(np_slope[:,np.newaxis], x.shape[3], axis=1)
  res_slope = np.repeat(res_slope[:,np.newaxis], x.shape[2], axis=1)
  res_slope = np.repeat(res_slope[:,np.newaxis], x.shape[1], axis=1)
  
  res_age = np.repeat(np.arange(N_YEAR)[np.newaxis], np_age.size, axis=0) + np_age[:,np.newaxis]
  res_age = np.repeat(res_age[:,:,np.newaxis], x.shape[-2], axis=-1)
  res_age = np.repeat(res_age[:,np.newaxis], x.shape[1], axis=1)
  res_age = np.expand_dims(res_age, axis=-1)
  
  # scale age
  res_age = res_age / 500.
  
  # concat
  res = np.concatenate([x, res_age.astype(np.float32)], axis=-1)
  res = np.concatenate([res, res_mean.astype(np.float32)], axis=-1)
  res = np.concatenate([res, res_slope.astype(np.float32)], axis=-1)
  
  return res

def addAgeTriplet_TF(x, np_age, np_mean, np_slope):
  '''
  Add age triplet
    x: [N_YEAR, N_AGE, N_T, N_BAND]
    np_age: [N_AGE]
    np_mean/np_slope: [N_AGE, N_BAND]
  '''
  
  # form data structure
  res_mean = tf.repeat(np_mean[tf.newaxis], x.shape[0], axis=0)
  res_mean = tf.repeat(res_mean[:,:,tf.newaxis], x.shape[2], axis=2)
  res_slope = tf.repeat(np_slope[tf.newaxis], x.shape[0], axis=0)
  res_slope = tf.repeat(res_slope[:,:,tf.newaxis], x.shape[2], axis=2)
  
  res_age = np.repeat(np.arange(N_YEAR)[np.newaxis], np_age.size, axis=0) + np_age[:,np.newaxis]
  res_age = tf.repeat(res_age[:,:,tf.newaxis], x.shape[-2], axis=-1)
  res_age = tf.expand_dims(res_age, axis=-1)
  res_age = tf.transpose(res_age, (1,0,2,3))
  
  # scale age
  res_age = tf.cast(res_age, dtype=tf.float32) / 500.
  
  # concat
  res = tf.concat([x, tf.cast(res_age, dtype=tf.float32)], axis=-1)
  res = tf.concat([res, tf.cast(res_mean, dtype=tf.float32)], axis=-1)
  res = tf.concat([res, tf.cast(res_slope, dtype=tf.float32)], axis=-1)
  
  return res

def addAgeTripletYear_TF(x, np_age, np_mean, np_slope):
  '''
  Add age triplet with yearly information
    x: [N_YEAR, N_AGE, N_T, N_BAND]
    np_age: [N_AGE]
    np_mean/np_slope: [N_AGE, N_YEAR, N_BAND]
  '''
  # form data structure
  np_mean = np.transpose(np_mean, (1,0,2))
  np_slope = np.transpose(np_slope, (1,0,2))
  
  res_mean = tf.repeat(np_mean[:,:,tf.newaxis], x.shape[2], axis=2)
  res_slope = tf.repeat(np_slope[:,:,tf.newaxis], x.shape[2], axis=2)
  
  res_age = np.repeat(np.arange(N_YEAR)[np.newaxis], np_age.size, axis=0) + np_age[:,np.newaxis]
  res_age = tf.repeat(res_age[:,:,tf.newaxis], x.shape[-2], axis=-1)
  res_age = tf.expand_dims(res_age, axis=-1)
  res_age = tf.transpose(res_age, (1,0,2,3))
  
  # scale age
  res_age = tf.cast(res_age, dtype=tf.float32) / 500.
  
  # concat
  res = tf.concat([x, tf.cast(res_age, dtype=tf.float32)], axis=-1)
  res = tf.concat([res, tf.cast(res_mean, dtype=tf.float32)], axis=-1)
  res = tf.concat([res, tf.cast(res_slope, dtype=tf.float32)], axis=-1)
  
  return res

def addAgeTriplet_perAge(x, np_age, np_mean, np_slope):
  ''' 
  For x with all ages, add age triplet to the last dimension of x
    x: [N_Sample, N_YEAR, N_T, N_BAND]
    np_age: [1]
    np_mean/np_slope: [N_BAND]
  '''
  # form data structure
  res_mean = np.repeat(np_mean[np.newaxis], x.shape[2], axis=0)
  res_mean = np.repeat(res_mean[np.newaxis], x.shape[1], axis=0)
  res_mean = np.repeat(res_mean[np.newaxis], x.shape[0], axis=0)
  res_slope = np.repeat(np_slope[np.newaxis], x.shape[2], axis=0)
  res_slope = np.repeat(res_slope[np.newaxis], x.shape[1], axis=0)
  res_slope = np.repeat(res_slope[np.newaxis], x.shape[0], axis=0)
  
  res_age = np.arange(N_YEAR)[np.newaxis] + np_age
  res_age = np.transpose(res_age, (1,0))
  res_age = np.repeat(res_age[:,np.newaxis], x.shape[-2], axis=1)
  res_age = np.repeat(res_age[np.newaxis], x.shape[0], axis=0)
  
  # scale age
  res_age = res_age / 500.
  
  # concat
  res = np.concatenate([x, res_age.astype(np.float32)], axis=-1)
  res = np.concatenate([res, res_mean.astype(np.float32)], axis=-1)
  res = np.concatenate([res, res_slope.astype(np.float32)], axis=-1)
  
  return res

def addAgeTripletYear_perAge(x, np_age, np_mean, np_slope):
  ''' 
  For x with all ages, add age triplet to the last dimension of x
    x: [N_Sample, N_YEAR, N_T, N_BAND]
    np_age: [1]
    np_mean/np_slope: [N_YEAR, N_BAND]
  '''
  # form data structure
  res_mean = np.repeat(np_mean[:,np.newaxis], x.shape[2], axis=1)
  res_mean = np.repeat(res_mean[np.newaxis], x.shape[0], axis=0)
  res_slope = np.repeat(np_slope[:,np.newaxis], x.shape[2], axis=1)
  res_slope = np.repeat(res_slope[np.newaxis], x.shape[0], axis=0)
  
  res_age = np.arange(N_YEAR)[np.newaxis] + np_age
  res_age = np.transpose(res_age, (1,0))
  res_age = np.repeat(res_age[:,np.newaxis], x.shape[-2], axis=1)
  res_age = np.repeat(res_age[np.newaxis], x.shape[0], axis=0)
  
  # scale age
  res_age = res_age / 500.
  
  # concat
  res = np.concatenate([x, res_age.astype(np.float32)], axis=-1)
  res = np.concatenate([res, res_mean.astype(np.float32)], axis=-1)
  res = np.concatenate([res, res_slope.astype(np.float32)], axis=-1)
  
  return res


def reshape_data_for_training(data_batch, N, num_year=N_YEAR, num_month=N_T, num_feature=N_FEA):
  
    data_batch_reshaped = tf.reshape(data_batch, [-1, num_year*num_month, num_feature])
    
    overlapped_data = []
    for i in range(num_year - N + 1):
        start_index = i * num_month
        end_index = (i + N) * num_month
        overlapped_data.append(data_batch_reshaped[:, start_index:end_index, :])

    overlapped_data_concat = tf.concat(overlapped_data, axis=0)
    
    return overlapped_data_concat
  
  

def convertAGE2D(inp, mask2d):
  out = np.full((inp.shape[0],mask2d.shape[0],mask2d.shape[1],inp.shape[-1]),
                np.nan)

  #  (492, 54152, 7) -> (492, 360, 720, 7)
  for i in range(out.shape[0]): 
    for k in range(out.shape[-1]): 
      out[i,:,:,k][mask2d] = inp[i,:,k]

  return out


def formGeo(x, x_geo):
  assert x.shape[0] == x_geo.shape[0]
  
  res = np.repeat(x_geo[:,np.newaxis], x.shape[2], axis=1)
  res = np.repeat(res[:,np.newaxis], x.shape[1], axis=1)
  
  res = np.concatenate([x, res], axis=-1)
  
  return res


def eval(model, path_test, path_out, age_tri, data_stat, age_used_idx, isPred2Input=True, isState=False, isTrain=False, isOutput=True, isDup=True, y_lim=[0,3.5]):
  print('--- Start evaluation ---')
  
  [tri_age, tri_mean, tri_slope] = age_tri
  [x_mean, x_std, y_mean, y_std] = data_stat
  [age_start_idx, age_end_idx] = age_used_idx
  
  print(f'  Target age: {tri_age}')  
  
  #--- Loading data ---  
  data_train = np.load(path_test)
  if not isTrain:
    x_name = 'x_test'
    y_name = 'y_test' 
  else: 
    x_name = 'x_train'
    y_name = 'y_train'

  x_test = normData(data_train[x_name], x_mean, x_std)
  y_test = normData(data_train[y_name], y_mean, y_std)
  
  y_test = y_test[age_start_idx:age_end_idx]

  del data_train
  gc.collect()


  #--- Preprocessing ---
  # duplicate annual prediction
  if isDup: 
    y_test = dupMonth(y_test, dup_idx=11)

  age_mean = normData(tri_mean, y_mean, y_std)
  age_slope = normData(tri_slope, y_mean, y_std)

  t_start = time.time()
  ls_rmse_all = []
  ls_pred_all = []
  for cur_age in tqdm(range(tri_age.size)):

    # append init
    cur_x = addInitY2X(x_test, y_test[cur_age:cur_age+1])
    cur_x = cur_x[0]
    cur_y = y_test[cur_age,:,1:,-1]
    # logging.info(cur_x.shape, cur_y.shape) # (852, 40, 12, 143) (852, 40, 7)

    # add age
    cur_x = addAgeTriplet_perAge(cur_x, tri_age[cur_age:(cur_age+1)], 
                            age_mean[cur_age], age_slope[cur_age])

    # predict
    ls_pred = []
    ls_rmse = []
    x_shape = cur_x.shape
    for i in range(x_shape[1]):
      tmp_x = cur_x[:,i].copy()
      if i>0:
        if isPred2Input:
          # replace init tree height with prediction
          tmp_x[:,:,N_CONS:(N_CONS+N_OUT)] = np.repeat(ls_pred[-1][:,np.newaxis,:], N_T, axis=1)
        
        if isState:
          cur_pred, h_state, c_state = model(tmp_x, h_state, c_state, training=False)
        else: 
          cur_pred = model(tmp_x, training=False)
          
      else:
        
        if isState:
          cur_pred, h_state, c_state = model(tmp_x, training=False)
        else: 
          cur_pred = model(tmp_x, training=False)
      
      cur_pred = cur_pred.numpy()
  
      # post-processing
      zero_idx = invNormData(tmp_x[:,-1,N_CONS],y_mean[0],y_std[0]) < 1
      # zero_idx = invNormData(cur_y[:,i,0],y_mean[0],y_std[0]) < 1
      cur_pred[zero_idx,0] = normData(np.array([0.],dtype=np.float32),y_mean[0],y_std[0])

      # save
      ls_pred.append(cur_pred)
      ls_rmse.append(mean_squared_error(invNormData(cur_y[:,i],y_mean,y_std), 
                                        invNormData(ls_pred[-1],y_mean,y_std), 
                                        multioutput='raw_values', squared=False))

    ls_rmse_all.append(np.array(ls_rmse))
    ls_pred_all.append(np.array(ls_pred))
    
    del cur_x, cur_y, tmp_x, cur_pred
    gc.collect()
    
  ls_rmse_all = np.array(ls_rmse_all)
  ls_pred_all = np.array(ls_pred_all)
  if isOutput:
    np.save(path_out+'_pred.npy', ls_pred_all)

    pd_acc = convertRMSE2Table(ls_rmse_all, tri_age)
    pd_acc.to_csv(path_out+'_acc_details.csv', index=False)
  print('  Spent {:.1f} min for evaluation.\n'.format((time.time()-t_start)/60))
  
  # plot age
  plotAge(ls_rmse_all, tri_age, isOutput, path_out, y_lim)
  
  return [y_test, ls_pred_all, ls_rmse_all]


def eval_Nyear(model, path_test, path_out, age_tri, data_stat, age_used_idx, num_year_segment, isPred2Input=True, isState=False, isTrain=False, isOutput=True, isDup=True, y_lim=[0,3.5]):
  print('--- Start evaluation ---')
  
  [tri_age, tri_mean, tri_slope] = age_tri
  [x_mean, x_std, y_mean, y_std] = data_stat
  [age_start_idx, age_end_idx] = age_used_idx
  
  print(f'  Target age: {tri_age}')  
  
  #--- Loading data ---  
  data_train = np.load(path_test)
  if not isTrain:
    x_name = 'x_test'
    y_name = 'y_test' 
  else: 
    x_name = 'x_train'
    y_name = 'y_train'

  x_test = normData(data_train[x_name], x_mean, x_std)
  y_test = normData(data_train[y_name], y_mean, y_std)
  
  y_test = y_test[age_start_idx:age_end_idx]

  del data_train
  gc.collect()


  #--- Preprocessing ---
  # duplicate annual prediction
  if isDup: 
    y_test = dupMonth(y_test, dup_idx=11)

  age_mean = normData(tri_mean, y_mean, y_std)
  age_slope = normData(tri_slope, y_mean, y_std)


  t_start = time.time()
  ls_rmse_all = []
  ls_pred_all = []
  for cur_age in tqdm(range(tri_age.size)):

    # append init
    cur_x = addInitY2X(x_test, y_test[cur_age:cur_age+1])
    cur_x = cur_x[0]
    cur_y = y_test[cur_age,:,1:,-1]
    # logging.info(cur_x.shape, cur_y.shape) # (852, 40, 12, 143) (852, 40, 7)

    # add age
    cur_x = addAgeTriplet_perAge(cur_x, tri_age[cur_age:(cur_age+1)], 
                            age_mean[cur_age], age_slope[cur_age])
    
  
    cur_x = np.reshape(cur_x, (-1,int(N_YEAR/num_year_segment),N_T*num_year_segment,N_FEA))
    cur_y = cur_y[:,(num_year_segment-1)::num_year_segment]

    # predict
    ls_pred = []
    ls_rmse = []
    x_shape = cur_x.shape
    for i in range(x_shape[1]):
      tmp_x = cur_x[:,i].copy()
      if i>0:
        if isPred2Input:
          # replace init tree height with prediction
          tmp_x[:,:,N_CONS:(N_CONS+N_OUT)] = np.repeat(ls_pred[-1][:,np.newaxis,:], N_T*num_year_segment, axis=1)
        
        if isState:
          cur_pred, h_state, c_state = model(tmp_x, h_state, c_state, training=False)
        else: 
          cur_pred = model(tmp_x, training=False)
          
      else:
        
        if isState:
          cur_pred, h_state, c_state = model(tmp_x, training=False)
        else: 
          cur_pred = model(tmp_x, training=False)
      
      cur_pred = cur_pred.numpy()
  
      # post-processing
      zero_idx = invNormData(tmp_x[:,-1,N_CONS],y_mean[0],y_std[0]) < 1
      # zero_idx = invNormData(cur_y[:,i,0],y_mean[0],y_std[0]) < 1
      cur_pred[zero_idx,0] = normData(np.array([0.],dtype=np.float32),y_mean[0],y_std[0])

      # save
      ls_pred.append(cur_pred)
      ls_rmse.append(mean_squared_error(invNormData(cur_y[:,i],y_mean,y_std), 
                                        invNormData(ls_pred[-1],y_mean,y_std), 
                                        multioutput='raw_values', squared=False))

    ls_rmse_all.append(np.array(ls_rmse))
    ls_pred_all.append(np.array(ls_pred))
    
    del cur_x, cur_y, tmp_x, cur_pred
    gc.collect()
    
  ls_rmse_all = np.array(ls_rmse_all)
  ls_pred_all = np.array(ls_pred_all)
  if isOutput:
    np.save(path_out+'_pred.npy', ls_pred_all)

    pd_acc = convertRMSE2Table(ls_rmse_all, tri_age)
    pd_acc.to_csv(path_out+'_acc_details.csv', index=False)
  print('  Spent {:.1f} min for evaluation.\n'.format((time.time()-t_start)/60))
  
  # plot age
  plotAge(ls_rmse_all, tri_age, isOutput, path_out, y_lim)
  
  return [y_test, ls_pred_all, ls_rmse_all]



def eval_multistep(model, path_test, path_out, age_tri, data_stat, age_used_idx, isPred2Input=True, isState=False, isTrain=False, isOutput=True, isDup=True, y_lim=[0,3.5]):
  print('--- Start evaluation ---')
  
  [tri_age, tri_mean, tri_slope] = age_tri
  [x_mean, x_std, y_mean, y_std] = data_stat
  [age_start_idx, age_end_idx] = age_used_idx
  
  print(f'  Target age: {tri_age}')  
  
  #--- Loading data ---  
  data_train = np.load(path_test)
  if not isTrain:
    x_name = 'x_test'
    y_name = 'y_test' 
  else: 
    x_name = 'x_train'
    y_name = 'y_train'

  x_test = normData(data_train[x_name], x_mean, x_std)
  y_test = normData(data_train[y_name], y_mean, y_std)
  
  y_test = y_test[age_start_idx:age_end_idx]

  del data_train
  gc.collect()


  #--- Preprocessing ---
  # duplicate annual prediction
  if isDup: 
    y_test = dupMonth(y_test, dup_idx=11)

  age_mean = normData(tri_mean, y_mean, y_std)
  age_slope = normData(tri_slope, y_mean, y_std)

  t_start = time.time()
  ls_rmse_all = []
  ls_pred_all = []
  for cur_age in tqdm(range(tri_age.size)):

    # append init
    cur_x = addInitY2X(x_test, y_test[cur_age:cur_age+1])
    cur_x = cur_x[0]
    cur_y = y_test[cur_age,:,1:,-1]
    # logging.info(cur_x.shape, cur_y.shape) # (852, 40, 12, 143) (852, 40, 7)

    # add age
    cur_x = addAgeTriplet_perAge(cur_x, tri_age[cur_age:(cur_age+1)], 
                            age_mean[cur_age], age_slope[cur_age])

    # predict
    ls_pred = []
    ls_rmse = []
    x_shape = cur_x.shape
    for i in range(x_shape[1]):
      tmp_x = cur_x[:,i].copy()
      if i>0:
        if isPred2Input:
          # replace init tree height with prediction
          tmp_x[:,:,N_CONS:(N_CONS+N_OUT)] = np.repeat(ls_pred[-1][:,np.newaxis,:], N_T, axis=1)
        
        if isState:
          cur_pred, h_state, c_state = model(tmp_x, h_state, c_state, training=False)
        else: 
          cur_pred = model(tmp_x, training=False)
          
      else:
        
        if isState:
          cur_pred, h_state, c_state = model(tmp_x, training=False)
        else: 
          cur_pred = model(tmp_x, training=False)
      
      cur_pred = cur_pred.numpy()[:,-1]
  
      # post-processing
      zero_idx = invNormData(tmp_x[:,-1,N_CONS],y_mean[0],y_std[0]) < 1
      # zero_idx = invNormData(cur_y[:,i,0],y_mean[0],y_std[0]) < 1
      cur_pred[zero_idx,0] = normData(np.array([0.],dtype=np.float32),y_mean[0],y_std[0])

      # save
      ls_pred.append(cur_pred)
      ls_rmse.append(mean_squared_error(invNormData(cur_y[:,i],y_mean,y_std), 
                                        invNormData(ls_pred[-1],y_mean,y_std), 
                                        multioutput='raw_values', squared=False))

    ls_rmse_all.append(np.array(ls_rmse))
    ls_pred_all.append(np.array(ls_pred))
    
    del cur_x, cur_y, tmp_x, cur_pred
    gc.collect()
    
  ls_rmse_all = np.array(ls_rmse_all)
  ls_pred_all = np.array(ls_pred_all)
  if isOutput:
    np.save(path_out+'_pred.npy', ls_pred_all)

    pd_acc = convertRMSE2Table(ls_rmse_all, tri_age)
    pd_acc.to_csv(path_out+'_acc_details.csv', index=False)
  print('  Spent {:.1f} min for evaluation.\n'.format((time.time()-t_start)/60))
  
  # plot age
  plotAge(ls_rmse_all, tri_age, isOutput, path_out, y_lim)
  
  return [y_test, ls_pred_all, ls_rmse_all]


def eval_Nyear_multistep(model, path_test, path_out, age_tri, data_stat, age_used_idx, num_year_segment, isPred2Input=True, isState=False, isTrain=False, isOutput=True, isDup=True, y_lim=[0,3.5]):
  print('--- Start evaluation ---')
  
  [tri_age, tri_mean, tri_slope] = age_tri
  [x_mean, x_std, y_mean, y_std] = data_stat
  [age_start_idx, age_end_idx] = age_used_idx
  
  print(f'  Target age: {tri_age}')  
  
  #--- Loading data ---  
  data_train = np.load(path_test)
  if not isTrain:
    x_name = 'x_test'
    y_name = 'y_test' 
  else: 
    x_name = 'x_train'
    y_name = 'y_train'

  x_test = normData(data_train[x_name], x_mean, x_std)
  y_test = normData(data_train[y_name], y_mean, y_std)
  
  y_test = y_test[age_start_idx:age_end_idx]

  del data_train
  gc.collect()


  #--- Preprocessing ---
  # duplicate annual prediction
  if isDup: 
    y_test = dupMonth(y_test, dup_idx=11)

  age_mean = normData(tri_mean, y_mean, y_std)
  age_slope = normData(tri_slope, y_mean, y_std)


  t_start = time.time()
  ls_rmse_all = []
  ls_pred_all = []
  for cur_age in tqdm(range(tri_age.size)):

    # append init
    cur_x = addInitY2X(x_test, y_test[cur_age:cur_age+1])
    cur_x = cur_x[0]
    cur_y = y_test[cur_age,:,1:,-1]
    # logging.info(cur_x.shape, cur_y.shape) # (852, 40, 12, 143) (852, 40, 7)

    # add age
    cur_x = addAgeTriplet_perAge(cur_x, tri_age[cur_age:(cur_age+1)], 
                            age_mean[cur_age], age_slope[cur_age])
    
  
    cur_x = np.reshape(cur_x, (-1,int(N_YEAR/num_year_segment),N_T*num_year_segment,N_FEA))
    cur_y = cur_y[:,(num_year_segment-1)::num_year_segment]

    # predict
    ls_pred = []
    ls_rmse = []
    x_shape = cur_x.shape
    for i in range(x_shape[1]):
      tmp_x = cur_x[:,i].copy()
      if i>0:
        if isPred2Input:
          # replace init tree height with prediction
          tmp_x[:,:,N_CONS:(N_CONS+N_OUT)] = np.repeat(ls_pred[-1][:,np.newaxis,:], N_T*num_year_segment, axis=1)
        
        if isState:
          cur_pred, h_state, c_state = model(tmp_x, h_state, c_state, training=False)
        else: 
          cur_pred = model(tmp_x, training=False)
          
      else:
        
        if isState:
          cur_pred, h_state, c_state = model(tmp_x, training=False)
        else: 
          cur_pred = model(tmp_x, training=False)
      
      cur_pred = cur_pred.numpy()[:,-1]
  
      # post-processing
      zero_idx = invNormData(tmp_x[:,-1,N_CONS],y_mean[0],y_std[0]) < 1
      # zero_idx = invNormData(cur_y[:,i,0],y_mean[0],y_std[0]) < 1
      cur_pred[zero_idx,0] = normData(np.array([0.],dtype=np.float32),y_mean[0],y_std[0])

      # save
      ls_pred.append(cur_pred)
      ls_rmse.append(mean_squared_error(invNormData(cur_y[:,i],y_mean,y_std), 
                                        invNormData(ls_pred[-1],y_mean,y_std), 
                                        multioutput='raw_values', squared=False))

    ls_rmse_all.append(np.array(ls_rmse))
    ls_pred_all.append(np.array(ls_pred))
    
    del cur_x, cur_y, tmp_x, cur_pred
    gc.collect()
    
  ls_rmse_all = np.array(ls_rmse_all)
  ls_pred_all = np.array(ls_pred_all)
  if isOutput:
    np.save(path_out+'_pred.npy', ls_pred_all)

    pd_acc = convertRMSE2Table(ls_rmse_all, tri_age)
    pd_acc.to_csv(path_out+'_acc_details.csv', index=False)
  print('  Spent {:.1f} min for evaluation.\n'.format((time.time()-t_start)/60))
  
  # plot age
  plotAge(ls_rmse_all, tri_age, isOutput, path_out, y_lim)
  
  return [y_test, ls_pred_all, ls_rmse_all]



def eval_old(model, path_test, path_out, age_tri, data_stat, age_used_idx, isState=False, isDup=True, isOutput=True, y_lim=[0,3.5]):
  # worked before 4/20/2024
  print('--- Start evaluation ---')
  
  [tri_age, tri_mean, tri_slope] = age_tri
  [x_mean, x_std, y_mean, y_std] = data_stat
  [age_start_idx, age_end_idx] = age_used_idx
  
  print(f'  Target age: {tri_age}')  
  
  #--- Loading data ---  
  data_train = np.load(path_test)
  x_test = normData(data_train['x_test'], x_mean, x_std)
  y_test = normData(data_train['y_test'], y_mean, y_std)
  
  y_test = y_test[age_start_idx:age_end_idx]

  del data_train
  gc.collect()


  #--- Preprocessing ---
  # duplicate annual prediction
  if isDup: 
    y_test = dupMonth(y_test, dup_idx=11)

  age_mean = normData(tri_mean, y_mean, y_std)
  age_slope = normData(tri_slope, y_mean, y_std)

  x_shape = x_test.shape


  t_start = time.time()
  ls_rmse_all = []
  ls_pred_all = []
  for cur_age in tqdm(range(tri_age.size)):

    # append init
    cur_x = addInitY2X(x_test, y_test[cur_age:cur_age+1])
    cur_x = cur_x[0]
    cur_y = y_test[cur_age,:,1:,-1]
    # logging.info(cur_x.shape, cur_y.shape) # (852, 40, 12, 143) (852, 40, 7)

    # add age
    cur_x = addAgeTriplet_perAge(cur_x, tri_age[cur_age:(cur_age+1)], 
                            age_mean[cur_age], age_slope[cur_age])

    # predict
    ls_pred = []
    ls_rmse = []
    for i in range(x_shape[1]):
      tmp_x = cur_x[:,i].copy()
      if i>0:
        # replace init tree height with prediction
        tmp_x[:,:,N_CONS:(N_CONS+N_OUT)] = np.repeat(ls_pred[-1][:,np.newaxis,:], N_T, axis=1)
        
        if isState:
          cur_pred, h_state, c_state = model(tmp_x, h_state, c_state, training=False)
        else: 
          cur_pred = model(tmp_x, training=False)
          
      else:
        
        if isState:
          cur_pred, h_state, c_state = model(tmp_x, training=False)
        else: 
          cur_pred = model(tmp_x, training=False)
      
      cur_pred = cur_pred.numpy()
  
      # post-processing
      zero_idx = invNormData(tmp_x[:,-1,N_CONS],y_mean[0],y_std[0]) < 1
      # zero_idx = invNormData(cur_y[:,i,0],y_mean[0],y_std[0]) < 1
      cur_pred[zero_idx,0] = normData(np.array([0.],dtype=np.float32),y_mean[0],y_std[0])

      # save
      ls_pred.append(cur_pred)
      ls_rmse.append(mean_squared_error(invNormData(cur_y[:,i],y_mean,y_std), 
                                        invNormData(ls_pred[-1],y_mean,y_std), 
                                        multioutput='raw_values', squared=False))

    ls_rmse_all.append(np.array(ls_rmse))
    ls_pred_all.append(np.array(ls_pred))
    
    del cur_x, cur_y, tmp_x, cur_pred
    gc.collect()
    
  ls_rmse_all = np.array(ls_rmse_all)
  ls_pred_all = np.array(ls_pred_all)
  if isOutput:
    np.save(path_out+'_pred.npy', ls_pred_all)

    pd_acc = convertRMSE2Table(ls_rmse_all, tri_age)
    pd_acc.to_csv(path_out+'_acc_details.csv', index=False)
  print('  Spent {:.1f} min for evaluation.\n'.format((time.time()-t_start)/60))


  # plot age
  plotAge(ls_rmse_all, tri_age, isOutput, path_out, y_lim)




def eval_AAO(model, path_test, path_out, age_tri, data_stat, age_used_idx, isTrain=False, isOutput=True, isDup=True, y_lim=[0,3.5]):
  '''
  Evaluate model to forecast All At Once (AAO) by just given the very begining initial condition
  '''
  
  [tri_age, tri_mean, tri_slope] = age_tri
  [x_mean, x_std, y_mean, y_std] = data_stat
  [age_start_idx, age_end_idx] = age_used_idx

  print(f'  Target age: {tri_age}')  

  #--- Loading data ---  
  data_train = np.load(path_test)
  if not isTrain:
    x_name = 'x_test'
    y_name = 'y_test' 
  else: 
    x_name = 'x_train'
    y_name = 'y_train'

  x_test = normData(data_train[x_name], x_mean, x_std)
  y_test = normData(data_train[y_name], y_mean, y_std)

  y_test = y_test[age_start_idx:age_end_idx]

  del data_train
  gc.collect()


  #--- Preprocessing ---
  # duplicate annual prediction
  if isDup: 
    y_test = dupMonth(y_test, dup_idx=11)

  age_mean = normData(tri_mean, y_mean, y_std)
  age_slope = normData(tri_slope, y_mean, y_std)

  t_start = time.time()
  ls_rmse_all = []
  ls_pred_all = []
  for cur_age in tqdm(range(tri_age.size)):

    # append init
    cur_x = addInitY2X(x_test, y_test[cur_age:cur_age+1])
    cur_x = cur_x[0]
    cur_y = y_test[cur_age,:,1:,-1]
    # logging.info(cur_x.shape, cur_y.shape) # (852, 40, 12, 143) (852, 40, 7)

    # add age
    cur_x = addAgeTriplet_perAge(cur_x, tri_age[cur_age:(cur_age+1)], 
                            age_mean[cur_age], age_slope[cur_age])

    # predict 
    cur_x = cur_x.reshape(-1,N_YEAR*N_T,N_FEA)
    cur_pred = model(cur_x, training=False)
    ls_rmse = []
    for i in range(7): 
      cur_rmse = mean_squared_error(invNormData(cur_y[:,:,i],y_mean[i],y_std[i]),
                                    invNormData(cur_pred[:,:,i],y_mean[i],y_std[i]), 
                                    multioutput='raw_values', squared=False)
      ls_rmse.append(cur_rmse)


    ls_rmse_all.append(np.array(ls_rmse))
    ls_pred_all.append(cur_pred)
    
    del cur_x, cur_y, cur_pred
    gc.collect()
    
  ls_rmse_all = np.array(ls_rmse_all)
  ls_rmse_all = np.transpose(ls_rmse_all, (0,2,1))
  ls_pred_all = np.array(ls_pred_all)
  if isOutput:
    np.save(path_out+'_pred.npy', ls_pred_all)

    pd_acc = convertRMSE2Table(ls_rmse_all, tri_age)
    pd_acc.to_csv(path_out+'_acc_details.csv', index=False)
  print('  Spent {:.1f} min for evaluation.\n'.format((time.time()-t_start)/60))

  # plot age
  plotAge(ls_rmse_all, tri_age, isOutput, path_out, y_lim)




def eval_glob(model, dir_root_int, path_xy, path_out, age_tri, data_stat, isDup=True, isOutput=True):   
  print('--- Start evaluation ---')
  
  t_start = time.time()
  
  [tri_age, tri_mean, tri_slope] = age_tri
  [x_mean, x_std, y_mean, y_std] = data_stat
  
  if not os.path.exists(path_out):
    os.makedirs(path_out)

  age = path_xy[1].split('glob_Y')[1].split('.')[0]
  print(f'  Target age: {age}')

  n_split = 2
  for sub_split in range(n_split): 
    #--- Loading data ---
    x_test = np.load(path_xy[0])
    y_test = np.load(path_xy[1])
    
    lt = x_test.shape[0]//n_split*sub_split
    rt = x_test.shape[0]//n_split*(sub_split+1)
    if sub_split == n_split-1:
      rt = x_test.shape[0]
    cur_x = x_test[lt:rt]
    cur_y = y_test[lt:rt]
    
    del x_test, y_test
    gc.collect()

    cur_x = normData(cur_x, x_mean, x_std)
    cur_y = normData(cur_y, y_mean, y_std)

    cur_y = np.expand_dims(cur_y, axis=0) # consistent with the previous eval()

    #--- Preprocessing ---
    # duplicate annual prediction
    if isDup: 
      cur_y = dupMonth(cur_y, dup_idx=11)

    age_mean = normData(tri_mean, y_mean, y_std)
    age_slope = normData(tri_slope, y_mean, y_std)

    x_shape = cur_x.shape
    

    # append init
    cur_x = addInitY2X(cur_x, cur_y)

    cur_x = cur_x[0]
    cur_y = cur_y[0,:,1:,-1]
    # logging.info(cur_x.shape, cur_y.shape) # (852, 40, 12, 143) (852, 40, 7)

    # add age
    cur_x = addAgeTriplet_perAge(cur_x, tri_age[tri_age == int(age)], 
                            age_mean[tri_age == int(age)][0], 
                            age_slope[tri_age == int(age)][0])

    # predict
    ls_pred = []
    for i in tqdm(range(x_shape[1])):
      tmp_x = cur_x[:,i].copy()
      if i>0:
        # replace init tree height with prediction
        tmp_x[:,:,N_CONS:(N_CONS+N_OUT)] = np.repeat(ls_pred[-1][:,np.newaxis,:], N_T, axis=1)

      # ls_pred.append(model(tmp_x, training=False))
      cur_pred = model(tmp_x, training=False).numpy()
  
      # post-processing
      zero_idx = invNormData(tmp_x[:,-1,N_CONS],y_mean[0],y_std[0]) < 1
      cur_pred[zero_idx,0] = normData(np.array([0.],dtype=np.float32),y_mean[0],y_std[0])

      # save
      ls_pred.append(cur_pred)
      
    ls_pred = np.array(ls_pred)
    ls_pred = invNormData(ls_pred,y_mean,y_std)

    if isOutput:
      np.save(os.path.join(path_out,'age'+age+'_pred_sub'+str(sub_split)+'.npy'), ls_pred)
      
    del cur_x, cur_y, tmp_x, cur_pred
    gc.collect()
      
  # save file
  ls_out = glob(os.path.join(path_out,'age'+age+'_pred_sub*.npy'))
  ls_out.sort()

  res_out = []
  for cur_out in ls_out:
    res_out.append(np.load(cur_out))
  res_out = np.concatenate(res_out, axis=1)
  np.save(os.path.join(path_out,'age'+age+'_pred.npy'), res_out)

  for path in ls_out:
    try:
      os.remove(path)
      print(f"\tFile {path} has been deleted successfully.")
    except FileNotFoundError:
      print(f"\tFile {path} was not found.")
    except Exception as e:
      print(f"\tAn error occurred while deleting {path}: {e}")
  
        
  # # output netcdf
  # mask_glob = np.load(os.path.join(dir_root_int,'Mask_all.npy'))
  # convertNPY2NC(res_out, mask_glob).to_netcdf(os.path.join(path_out,'age'+age+'_pred.nc'))
  
  
  # # output difference
  # # the differences are positive due to the convertNPY2NC function
  # true_out = np.load(os.path.join(dir_root_int,'data_eval_global','glob_Y'+age+'.npy'))
  # true_out = true_out[:,1:,-1] # use Dec month from 2nd years
  # true_out = np.transpose(true_out, (1,0,2))

  # diff_out = res_out - true_out
  # convertNPY2NC(diff_out, mask_glob).to_netcdf(os.path.join(path_out,'age'+age+'_pred-true.nc'))


  print('  Spent {:.1f} min for evaluation.\n'.format((time.time()-t_start)/60))

  return res_out