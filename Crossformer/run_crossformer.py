#!/usr/bin/env python
# coding: utf-8

# In[6]:

import logging
import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import time 
import gc
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from cross_exp.exp_crossformer import Exp_crossformer_deeped
from utils.tools import dotdict, string_split


# In[2]:


args = dotdict()

# Data parameters
args.data = 'deeped'  # data
args.root_path = os.path.join('..', 'deeped', 'datasets', 'global')  # root path of the data file
args.data_path = 'res_train4_test8.npz' # data file
args.stat_path = 'data_stats.npz' # stat file
args.asi = 0
args.aei = 15
args.add_noise = True
args.noise_std = 1e-4

args.data_split = '0.7,0.1,0.2'  # train/val/test split, can be ratio or number
args.checkpoints = './checkpoints/'  # location to store model checkpoints

# Input/Output lengths
args.in_len = 12  # input MTS length (T)
args.lab_len = 1  # initial label length in decoder
args.out_len = 1  # output MTS length (τ)
args.seg_len = 6  # segment length (L_seg)
args.win_size = 2  # window size for segment merge
args.factor = 10  # num of routers in Cross-Dimension Stage of TSA (c)

# Model parameters
args.data_dim_int = 136  # [Updated] Number of dimensions of the MTS data (D)
args.data_dim_out = 7  # [Updated] Number of dimensions of the MTS data (D)
args.d_model = 128  # dimension of hidden states (d_model)
args.d_ff = 512  # dimension of MLP in transformer
args.n_heads = 4  # num of heads
args.e_layers = 2  # num of encoder layers (N)
args.dropout = 0.05  # dropout

# Baseline
args.baseline = False  # whether to use mean of past series as baseline for prediction

# Training parameters
args.num_workers = 0  # data loader num workers
args.batch_size = 1200  # batch size of train input data
args.train_epochs = 20  # train epochs
args.patience = 3  # early stopping patience
args.learning_rate = 1e-4  # optimizer initial learning rate
args.lradj = 'type1'  # adjust learning rate
args.itr = 1  # experiments times

# Prediction saving
args.save_pred = False  # whether to save the predicted future MTS

# GPU parameters
args.use_gpu = True  # use gpu
args.gpu = 0  # gpu
args.use_multi_gpu = False  # use multiple gpus
args.devices = '0,1,2,3'  # device ids of multiple gpus


# In[3]:


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    print(args.gpu)

data_parser = {
    'ETTh1':{'data':'ETTh1.csv', 'data_dim':7, 'split':[12*30*24, 4*30*24, 4*30*24]},
    'ETTm1':{'data':'ETTm1.csv', 'data_dim':7, 'split':[4*12*30*24, 4*4*30*24, 4*4*30*24]},
    'WTH':{'data':'WTH.csv', 'data_dim':12, 'split':[28*30*24, 10*30*24, 10*30*24]},
    'ECL':{'data':'ECL.csv', 'data_dim':321, 'split':[15*30*24, 3*30*24, 4*30*24]},
    'ILI':{'data':'national_illness.csv', 'data_dim':7, 'split':[0.7, 0.1, 0.2]},
    'Traffic':{'data':'traffic.csv', 'data_dim':862, 'split':[0.7, 0.1, 0.2]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.data_dim = data_info['data_dim']
    args.data_split = data_info['split']
else:
    args.data_split = string_split(args.data_split)

print('Args in experiment:')
print(args)


# In[13]:
#-- set up logging

Exp = Exp_crossformer_deeped

ii = 4
# setting record of experiments
setting = 'Crossformer_{}_il{}_ol{}_sl{}_win{}_fa{}_dm{}_nh{}_el{}_itr{}_noise{}'.format(args.data, 
            args.in_len, args.out_len, args.seg_len, args.win_size, args.factor,
            args.d_model, args.n_heads, args.e_layers, ii, int(args.add_noise))

exp = Exp(args) # set experiments


# In[ ]:

logging.basicConfig(filename='results/'+setting+'.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')
logging.info('\n==========Train Details=============')




# print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
# exp.train(setting, logging)

# print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
# exp.test(setting)


path = os.path.join(exp.args.checkpoints, setting)
best_model_path = path+'/'+'checkpoint.pth'
exp.model.load_state_dict(torch.load(best_model_path))
print('load model from:', best_model_path)

# In[7]:


# train_data, train_loader = exp._get_data(flag='train')

# seq_x, seq_y = train_data[0]

# for i, (batch_x,batch_y) in enumerate(train_loader):
#   pred, true = exp._process_one_batch(
#     train_data, batch_x, batch_y)
#   print('output:', pred.shape, true.shape)
#   break


# In[8]:


# Exp = Exp_crossformer_deeped

# for ii in range(args.itr):
#     # setting record of experiments
#     setting = 'Crossformer_{}_il{}_ol{}_sl{}_win{}_fa{}_dm{}_nh{}_el{}_itr{}'.format(args.data, 
#                 args.in_len, args.out_len, args.seg_len, args.win_size, args.factor,
#                 args.d_model, args.n_heads, args.e_layers, ii)

#     exp = Exp(args) # set experiments
#     print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
#     exp.train(setting)
    
#     print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#     exp.test(setting, args.save_pred)


# ## Prediction

# In[9]:


#-- resutl output path
DIR_INT = os.path.join('..', 'deeped', 'datasets', 'global')
# DIR_OUT = 'results'

# output bands
TREE_BAND = ['height', 'agb', 'soil', 'lai', 'gpp', 'npp', 'rh']


# parameters
N_T = 12 
N_YEAR = 40
N_AGE = 15
N_OUT = len(TREE_BAND)

N_CONS = 136 # number of atmos features
N_FEA = N_CONS + N_OUT + 15 # add age info

eval_filename = 'res_train4_test8.npz'
agetri_filename = 'age_triplet.npz'
stat_filename = 'data_stats.npz'

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

age_start_idx = args.asi
age_end_idx = args.aei

# select age
TRI_AGE = TRI_AGE[age_start_idx:age_end_idx]
TRI_MEAN = TRI_MEAN[age_start_idx:age_end_idx]
TRI_SLOPE = TRI_SLOPE[age_start_idx:age_end_idx]
N_AGE = TRI_AGE.size
print(f'Working on age: {TRI_AGE}')

age_tri = [TRI_AGE, TRI_MEAN, TRI_SLOPE]
data_stat = [X_MEAN, X_STD, Y_MEAN, Y_STD]


def normData(inp, mean, std): 
  # assert inp.shape[-1] == mean.size
  # assert std.size == mean.size
  
  return (inp-mean)/(std+1e-10)

def invNormData(inp, mean, std): 
  # assert inp.shape[-1] == mean.size
  # assert std.size == mean.size
  
  return inp*(std+1e-10)+mean

def dupMonth(y, dup_idx=11):
  # assert dup_idx < N_T
  
  return np.repeat(y[...,dup_idx:dup_idx+1,:], N_T, axis=-2)

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
  
  
def addInitY2X(x, y): 
  out = np.repeat(x[np.newaxis], y.shape[0], axis=0)
  return np.concatenate([out, y[:,:,:-1]], axis=-1)

[tri_age, tri_mean, tri_slope] = age_tri
[x_mean, x_std, y_mean, y_std] = data_stat

print(f'  Target age: {tri_age}')  
  

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


# predict using prediction
def eval(emulator, path_test, path_out, age_tri, data_stat, age_used_idx, isPred2Input=True, isTrain=False, isOutput=True, isDup=True, y_lim=[0,3.5]):
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
  
  print(f'X shape: {x_test.shape}, Y shape: {y_test.shape}')
  
  emulator.model.eval()


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
      tmp_x = cur_x[:,i].copy() # (B, 12, 158)
      tmp_y = cur_y[:,i].copy() # (B, 7)
      if i>0:
        if isPred2Input:
          # replace init tree height with prediction
          tmp_x[:,:,N_CONS:(N_CONS+N_OUT)] = np.repeat(ls_pred[-1][:,np.newaxis,:], N_T, axis=1)
      
      # use target only
      tmp_y = np.expand_dims(tmp_y, axis=1) # (B, 1, 7)
      tmp_y = np.concatenate([tmp_x[:, 0:1, 136:143], tmp_y], axis=1) # (B, 2, 7)
      tmp_x = tmp_x[:, :, :136] # (B, 12, 136)
      
      # # use age triplets
      # tmp_y = np.expand_dims(tmp_y, axis=1) # (B, 1, 7)
      # tmp_y1 = tmp_x[:, 0:1, 136:] # (batch*40, 1, 22)
      # tmp_y2 = np.concatenate([tmp_y, np.zeros((tmp_y.shape[0], 1, 15))], 2) # (batch*40, 1, 22)
      # tmp_y = np.concatenate([tmp_y1, tmp_y2], 1) # (batch*40, 2, 22)
      # tmp_x = tmp_x[:, :, :136] # (B, 12, 136)
          
      cur_pred, _ = emulator._process_one_batch(0, torch.tensor(tmp_x), torch.tensor(tmp_y))
      cur_pred = cur_pred.detach().cpu().numpy().squeeze()
  
      # post-processing
      zero_idx = invNormData(tmp_y[:,0,0],y_mean[0],y_std[0]) < 1
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


# In[ ]:


dir_out = './checkpoints/' + setting +'/'

# _ = eval(exp, 
#   os.path.join(args.root_path, args.data_path), 
#   dir_out+'test_pred', 
#   age_tri, data_stat, [age_start_idx,age_end_idx], 
#   isPred2Input=True, isTrain=False, isOutput=True, isDup=True, y_lim=[0,3.5])

# _ = eval(exp, 
#   os.path.join(args.root_path, args.data_path), 
#   dir_out+'test_true', 
#   age_tri, data_stat, [age_start_idx,age_end_idx], 
#   isPred2Input=False, isTrain=False, isOutput=True, isDup=True, y_lim=[0,3.5])




import xarray as xr
from glob import glob
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def eval_glob(emulator, dir_root_int, path_xy, path_out, age_tri, data_stat, isDup=True, isOutput=True):   
  print('--- Start evaluation ---')
  
  t_start = time.time()
  
  [tri_age, tri_mean, tri_slope] = age_tri
  [x_mean, x_std, y_mean, y_std] = data_stat
  
  if not os.path.exists(path_out):
    os.makedirs(path_out)

  age = path_xy[1].split('glob_Y')[1].split('.')[0]
  print(f'  Target age: {age}')

  emulator.model.eval() 
  with torch.no_grad():
    n_split = 8
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
        tmp_x = cur_x[:,i].copy() # (B, 12, 158)
        tmp_y = cur_y[:,i].copy() # (B, 7)
        if i>0:
          # replace init tree height with prediction
          tmp_x[:,:,N_CONS:(N_CONS+N_OUT)] = np.repeat(ls_pred[-1][:,np.newaxis,:], N_T, axis=1)      
        
        # use target only
        tmp_y = np.expand_dims(tmp_y, axis=1) # (B, 1, 7)
        tmp_y = np.concatenate([tmp_x[:, 0:1, 136:143], tmp_y], axis=1) # (B, 2, 7)
        tmp_x = tmp_x[:, :, :136] # (B, 12, 136)
        
        cur_pred, _ = emulator._process_one_batch(0, torch.tensor(tmp_x), torch.tensor(tmp_y))
        cur_pred = cur_pred.detach().cpu().numpy().squeeze()

        # post-processing
        zero_idx = invNormData(tmp_y[:,0,0],y_mean[0],y_std[0]) < 1
        cur_pred[zero_idx,0] = normData(np.array([0.],dtype=np.float32),y_mean[0],y_std[0])

        # save
        ls_pred.append(cur_pred)
        
      ls_pred = np.array(ls_pred)
      ls_pred = invNormData(ls_pred,y_mean,y_std)

      if isOutput:
        np.save(os.path.join(path_out,'age'+age+'_pred_sub'+str(sub_split)+'.npy'), ls_pred)
        
      del cur_x, cur_y, tmp_x, cur_pred
      gc.collect()
      torch.cuda.empty_cache()
      
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

def convertAGE2D(inp, mask2d):
  out = np.full((inp.shape[0],mask2d.shape[0],mask2d.shape[1],inp.shape[-1]),
                np.nan)

  #  (492, 54152, 7) -> (492, 360, 720, 7)
  for i in range(out.shape[0]): 
    for k in range(out.shape[-1]): 
      out[i,:,:,k][mask2d] = inp[i,:,k]

  return out


def convertNPY2NC(inp_npy, mask_glob): 
  res_out2d = convertAGE2D(inp_npy, mask_glob)

  raw_nc = xr.open_dataset(os.path.join(DIR_INT, 'data_eval_global', 'ED_restart-40.nc'))
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
  tmp_out = eval_glob(exp, DIR_INT, path_xy, dir_out+'_glob', age_tri, data_stat, isDup=True, isOutput=True)
  convertNPY2NC(tmp_out, mask_glob).to_netcdf(os.path.join(dir_out+'_glob','age'+path_y.split('glob_Y')[1].split('.')[0]+'_pred.nc'))
  

# evaluation
dir_out_eval = dir_out+'_glob'

ls_path_true = glob(os.path.join(DIR_INT, 'data_eval_global', 'glob_Y*.npy'))
ls_path_pred = glob(os.path.join(dir_out_eval, '*.npy'))
ls_path_true.sort()
ls_path_pred.sort()
print(f'Length of true and pred: {len(ls_path_true)}, {len(ls_path_pred)}')

# find test samples 
mask_glob = np.load(os.path.join(DIR_INT,'Mask_all.npy')).astype(int)
mask_train_test = np.load(os.path.join(DIR_INT,'train_test_mask.npy'))
mask_glob[mask_train_test==1] = 2
mask_glob = mask_glob[mask_glob>0]
test_site = mask_glob == 1


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


