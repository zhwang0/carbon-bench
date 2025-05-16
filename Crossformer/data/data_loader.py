import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler

import warnings
warnings.filterwarnings('ignore')



class Dataset_DeepED(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='data_train.npz', stat_path='stat.npz',
                 asi=0, aei=15, val_ratio=0.1,
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'pred']
        if flag == 'pred':
            flag = 'test'
        self.flag = flag
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.stat_path = stat_path
        self.asi = asi
        self.aei = aei
        self.val_ratio = val_ratio
         
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        
        self.__read_data__()

    def __read_data__(self):
        stat_raw = np.load(os.path.join(self.root_path, self.stat_path))
        self.x_mean = stat_raw['x_mean']
        self.x_std = stat_raw['x_std']
        self.y_mean = stat_raw['y_mean']
        self.y_std = stat_raw['y_std']
        
        data_raw = np.load(os.path.join(self.root_path, self.data_path))
        # x = (N, 40, 12, 136)
        # y = (age, N, 41, 12, 7)
        if self.set_type == 2: 
            data_x = data_raw[f'x_test'] 
            data_y = data_raw[f'y_test'] 
        else: 
            data_x = data_raw[f'x_train']
            data_y = data_raw[f'y_train']
        
        # only select last month as target 
        data_y = data_y[self.asi:self.aei, :, :, -1] # (age, N, 41, 7)
        
        # prepare inital and target pair 
        data_y = self.__sliding_window__(data_y, 2, 1, 2) # (age, N, 40, 2, 7)
        data_y = np.transpose(data_y, (1,0,2,3,4)) # (N, age, 40, 2, 7)
        
        # dup x for matching ages 
        data_x = np.repeat(data_x[:, np.newaxis, :, :, :], self.aei-self.asi, axis=1) # (N, age, 40, 12, 136)
            
        # normalize data
        data_x = self.__norm_data__(data_x, self.x_mean, self.x_std)
        data_y = self.__norm_data__(data_y, self.y_mean, self.y_std)
        
        # split train and val
        if self.set_type != 2:
            idx = np.arange(data_x.shape[0])
            np.random.shuffle(idx)
            idx_val = idx[:int(len(idx)*self.val_ratio)]
            idx_train = idx[int(len(idx)*self.val_ratio):]
            if self.set_type == 0:
                data_x = data_x[idx_train]
                data_y = data_y[idx_train]
            else:
                data_x = data_x[idx_val]
                data_y = data_y[idx_val]
        
        
        self.data_x = data_x.reshape(-1, data_x.shape[3], data_x.shape[4])
        self.data_y = data_y.reshape(-1, data_y.shape[3], data_y.shape[4]) 
        print(f'{self.flag} data size x={self.data_x.shape}, y={self.data_y.shape}')
        

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
    
        return seq_x, seq_y
    
    def __len__(self):
        return self.data_x.shape[0]
    
    def __sliding_window__(self, arr, window_size, stride, axis):
        # Ensure the axis is positive and within the valid range
        axis = axis if axis >= 0 else arr.ndim + axis
        if axis < 0 or axis >= arr.ndim:
            raise ValueError("Axis out of bounds")

        # Calculate the shape and strides for the sliding window
        new_shape = list(arr.shape)
        new_shape[axis] = (arr.shape[axis] - window_size) // stride + 1
        new_shape.insert(axis + 1, window_size)
        
        new_strides = list(arr.strides)
        new_strides.insert(axis + 1, new_strides[axis])
        new_strides[axis] = new_strides[axis] * stride

        # Create the sliding window view
        return np.lib.stride_tricks.as_strided(arr, shape=tuple(new_shape), strides=tuple(new_strides))
    
    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_DeepED_backup(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='data_train.npz', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'pred']
        if flag == 'pred':
            flag = 'test'
        self.flag = flag
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = f'data_{flag}.npz'
        self.__read_data__()

    def __read_data__(self):
        df_raw = np.load(os.path.join(self.root_path, 
                                      self.data_path))
        data_x = df_raw[f'x_{self.flag}'] # (batch, 40, 12, 158) 
        data_y = df_raw[f'y_{self.flag}'] # (batch, 40, 7)
        
        # --- only use 7 output features ---
        data_x = data_x.reshape(-1, data_x.shape[2], data_x.shape[3]) # (batch*40, 12, 158)
        data_y = data_y.reshape(-1, data_y.shape[2])
        data_y = np.expand_dims(data_y, axis=1) # (batch*40, 1, 7)
        data_y = np.concatenate([data_x[:, 0:1, 136:143], data_y], 1) # (batch*40, 2, 7)
        data_x = data_x[:, :, :136] # (batch*40, 12, 136)
        
        # # --- use age triplets ---
        # data_x = data_x.reshape(-1, data_x.shape[2], data_x.shape[3]) # (batch*40, 12, 158)
        # data_y = data_y.reshape(-1, data_y.shape[2])
        # data_y = np.expand_dims(data_y, axis=1) # (batch*40, 1, 7)
        # data_y1 = data_x[:, 0:1, 136:] # (batch*40, 1, 22)
        # # (batch*40, 1, 7) --> (batch*40, 1, 22)
        # data_y2 = np.concatenate([data_y, np.zeros((data_y.shape[0], 1, 15))], 2) 
        # data_y = np.concatenate([data_y1, data_y2], 1) # (batch*40, 2, 22)
        # data_x = data_x[:, :, :136] # (batch*40, 12, 136)
        
        
        # data_y = np.repeat(data_y[:, :, np.newaxis, :], 12, axis=2) # (batch, 40, 12, 7)
        # data_x = data_x.reshape(-1, data_y.shape[1]*data_y.shape[2], data_x.shape[-1]) # (batch, 480, 158)
        # data_y = data_y.reshape(-1, data_y.shape[1]*data_y.shape[2], data_y.shape[-1]) # (batch, 480, 7)
        
        self.data_x = data_x
        self.data_y = data_y
        
    
    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]

        return seq_x, seq_y
    
    def __len__(self):
        return self.data_x.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_MTS(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', flag='train', size=None, 
                  data_split = [0.7, 0.1, 0.2], scale=True, scale_statistic=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.in_len = size[0]
        self.out_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.scale = scale
        #self.inverse = inverse
        
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if (self.data_split[0] > 1):
            train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
        else:
            train_num = int(len(df_raw)*self.data_split[0]); 
            test_num = int(len(df_raw)*self.data_split[2])
            val_num = len(df_raw) - train_num - test_num; 
        border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
        border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean = self.scale_statistic['mean'], std = self.scale_statistic['std'])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.in_len
        r_begin = s_end
        r_end = r_begin + self.out_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.in_len- self.out_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)