import os 
import numpy as np

def convertCMP2Raw(inp, mask2d, inp_type):
    '''
    inp_type: first three letters of file suffix
    
    '''    
    if inp_type == 'clm':
        out = np.full((inp.shape[0],inp.shape[1],mask2d.shape[0],mask2d.shape[1],inp.shape[-1]),
                      np.nan)
        start_idx = 2
        
    elif inp_type == 'co2':
        out = np.full((inp.shape[0],inp.shape[1],mask2d.shape[0],mask2d.shape[1]),
                      np.nan)
        start_idx = 2

    elif inp_type == 'dst':
        out = np.full((inp.shape[0],mask2d.shape[0],mask2d.shape[1]),
                      np.nan)
        start_idx = 1
        
    elif inp_type == 'mch':
        out = np.full((inp.shape[0],inp.shape[1],mask2d.shape[0],mask2d.shape[1],inp.shape[-1]),
                      np.nan)
        start_idx = 2
        
    elif inp_type == 'soi':
        out = np.full((mask2d.shape[0],mask2d.shape[1],inp.shape[-1]),
                      np.nan)
        start_idx = 0

    elif inp_type == 'age':
        out = np.full((inp.shape[0],mask2d.shape[0],mask2d.shape[1],inp.shape[-1]),
                      np.nan)
        start_idx = 1

    else:
        print('Input Data Type is not support!') 
        return

    # copy data
    if start_idx == 0: 
        if len(out.shape) == 3: 
            # (54152, 7) -> (360, 720, 7)
            for k in range(out.shape[-1]): 
                out[...,k][mask2d] = inp[...,k]
        else: 
            out[mask2d] = inp

    elif start_idx == 1: 
        if len(out.shape) == 4: 
            #  (492, 54152, 7) -> (492, 360, 720, 7)
            for i in range(out.shape[0]): 
                for k in range(out.shape[-1]): 
                    out[i,:,:,k][mask2d] = inp[i,:,k]
                    
        else:
            #  (41, 54152) -> (41, 360, 720)
            for i in range(out.shape[0]): 
                out[i][mask2d] = inp[i]

    elif start_idx == 2: 
        if len(out.shape) == 5: 
            # (41, 12, 54152, 8) -> (41, 12, 360, 720, 8)
            for i in range(out.shape[0]): 
                for j in range(out.shape[1]): 
                    for k in range(out.shape[-1]): 
                        out[i,j,:,:,k][mask2d] = inp[i,j,:,k]

        else:
            #(41, 288, 54152) -> (41, 288, 360, 720)
            for i in range(out.shape[0]): 
                for j in range(out.shape[1]): 
                    out[i,j][mask2d] = inp[i,j]

    else: 
        print('Wrong start_idx! Please double check the codes.')

    return out
  
  
  
def generateTrainTestGrid(mask2d, step_trai, step_test):
  
  res = np.zeros_like(mask2d, dtype=int)
  
  # train
  for i in range(0, mask2d.shape[0], step_trai):
    for j in range(0, mask2d.shape[1], step_trai):
      res[i,j] = 1 
  
  # test
  for i in range(int(step_trai/2), mask2d.shape[0], step_test): 
    for j in range(int(step_trai/2), mask2d.shape[1], step_test):
      res[i,j] = 2 
      
  res = res * mask2d
      
  print('\n======= Grid-Like Train/Test Sample Sizes =======')
  print('Total_sample_size = ', np.sum(mask2d==1))
  print('Train_size = ', np.sum(res==1))
  print('Test_size = ', np.sum(res==2))

  return res


def extractTrainTest2dTo1d(inp2d, mask1d, inp_type):
    '''
    inp_type: first three letters of file suffix
    
    '''
    
    if inp_type == 'clm':
        out = inp2d.reshape(inp2d.shape[0],inp2d.shape[1],-1,inp2d.shape[-1])
        out = out[:,:,mask1d]
        
    elif inp_type == 'co2':
        out = inp2d.reshape(inp2d.shape[0],inp2d.shape[1],-1)
        out = out[:,:,mask1d]

    elif inp_type == 'dst':
        out = inp2d.reshape(inp2d.shape[0],-1)
        out = out[:,mask1d]

    elif inp_type == 'mch':
        out = inp2d.reshape(inp2d.shape[0],inp2d.shape[1],-1,inp2d.shape[-1])
        out = out[:,:,mask1d]

    elif inp_type == 'soi':
        out = inp2d.reshape(-1,inp2d.shape[-1])
        out = out[mask1d]

    elif inp_type == 'age':
        out = inp2d.reshape(inp2d.shape[0],-1,inp2d.shape[-1])
        out = out[:,mask1d]

    else:
        print('Input Data Type is not support!') 
        return

    return out



def generateTrainTest(inp_path, all_mask2d, train_test_mask2d):
  
  inp_type = os.path.basename(inp_path).split('_')[-1][:3]
  
  data_raw = np.load(inp_path)
  data2d = convertCMP2Raw(data_raw, all_mask2d, inp_type)
  
  res_tria = extractTrainTest2dTo1d(data2d, (train_test_mask2d==1).flatten(), inp_type)
  res_test = extractTrainTest2dTo1d(data2d, (train_test_mask2d==2).flatten(), inp_type)
  
  return res_tria, res_test



def shapeCLM(raw_clm): 
  res = np.transpose(raw_clm, (2,0,1,3))
  return res
  
def shapeCO2(raw_co2, N_YEAR=40, N_T=12): 
  res = np.transpose(raw_co2, (2,0,1))
  res = res.reshape(-1,N_YEAR,N_T,24)
  return res

def shapeDST(raw_dst): 
  res = np.repeat(raw_dst[:,:,np.newaxis], 12, axis=2)
  res = np.transpose(res, (1,0,2))
  res = np.expand_dims(res, axis=3)   
  return res
  
def shapeMCH(raw_mach, N_YEAR=40, N_T=12): 
  res = np.transpose(raw_mach, (2,0,1,3))
  res = res.reshape(-1,N_YEAR,N_T,24*4)
  res = res.reshape(-1,N_YEAR,N_T,24*4)
  return res

def shapeSOI(raw_soi, N_YEAR=40, N_T=12):  
  res = np.repeat(raw_soi[:,np.newaxis,:], N_T, axis=1)
  res = np.repeat(res[:,np.newaxis,:,:], N_YEAR, axis=1)
  return res
  
def shapeAGE(raw_age, N_YEAR=40, N_T=12): 
  res = np.transpose(raw_age, (1,0,2))
  res = res.reshape(res.shape[0],N_YEAR+1,N_T,res.shape[-1])
  return res

  
def concatList(ls_data):
  res = np.concatenate((ls_data[0], ls_data[1]), axis=-1)
  if len(ls_data) > 2: 
    for i in range(2, len(ls_data)): 
      res = np.concatenate([res, ls_data[i]], axis=-1)
  return res
  