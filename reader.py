# -*- coding: utf-8 -*- 
from __future__ import absolute_import
from __future__ import division

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame


def read_data(name, missing_flag):
    data = pd.read_excel(name).iloc[:,1 :6]
    columns_name = list(data.columns)
    ##replace space  with np.nan
    data.replace(to_replace=r'^\s*$',value=np.nan,regex=True,inplace=True)
    
    ##z-score
    norlizer = StandardScaler().fit(data)
    data = norlizer.transform(data)
    
    ##replace the nan with missing_flag
    data[np.isnan(data)] = missing_flag
    return data, columns_name, norlizer

def save_data(miss_data, prediction, epoch, norlizer, columns_name, missing_flag):
	###make sure the first row of you data is completed
	prediction = np.concatenate((miss_data[0,:].reshape(1,-1), np.array(prediction)), axis=0)
	imputation = miss_data.copy()
	assert imputation.shape == prediction.shape
	imputation[miss_data == missing_flag] = prediction[miss_data == missing_flag]
	
	###reverse the data
	imputation = norlizer.inverse_transform(imputation)
	
	###save it
	new_df = DataFrame(imputation, columns=columns_name)
	new_df.to_excel('imputed_data_epoch{0}.xlsx'.format(epoch), index=None)
    

def RMSE_Metric(data, pre, missing_flag):
    data = np.array(data).copy()
    pre = np.array(pre).copy()
    assert data.shape == pre.shape
    return np.sqrt(np.mean((data[ ~(data == missing_flag)] - pre[~(data == missing_flag)])**2))
    

def ptb_iterator(raw_data, batch_size, num_steps,dim):
    row = raw_data.shape[0]
    data_num = row-num_steps+1
    if batch_size>=(data_num-1) or ((data_num-1)%batch_size!=0) :
        raise ValueError("Error, decrease batch_size or num_steps")
    data2 = np.zeros([data_num, num_steps ,dim])
    for i in range(data_num):
        data2[i] = raw_data[i:i+num_steps,:]

    batch_len=(data_num-1)//batch_size
    for i in range(batch_len):
        x = data2[i*batch_size:(i+1)*batch_size,:,:]
        y = data2[i*batch_size+1:(i+1)*batch_size+1,:,:]
        yield (x,y)

if __name__ == '__main__':
    data, columns_name, norlizer = read_data('missing_data.xlsx', missing_flag=-128.0)
