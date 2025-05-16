import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from configs.constant_glob_age import *


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
  print('The best model is restored from:\n\t', best_ckp)
  return best_ckp


def plotAge(np_rmse, TRI_AGE, isOutput=True, path_out='', y_lim=[0,3.5]):
  # plot age
  plt.figure()
  fig_r = 3
  fig_c = 5
  ls_cor = ['r','g','b','c','y','m','k']
  fig, axes = plt.subplots(fig_r, fig_c, figsize=(10,7))
  for i in range(fig_r):
    for j in range(fig_c):
      if i*fig_c+j < np_rmse.shape[0]: 
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