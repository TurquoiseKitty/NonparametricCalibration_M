import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import os
import torch
import pandas as pd




def get_uci_data(data_name, dir_name = "Datasets/UCI_datasets"):
    
    data = np.loadtxt("{}/{}.txt".format(dir_name, data_name))
    x_al = data[:, :-1]
    y_al = data[:, -1].reshape(-1)

    return x_al, y_al

def normalize(data):
    
    normalizer = StandardScaler().fit(data)
    
    return normalizer.transform(data), normalizer



def common_processor_UCI(x, y, tr_re_va_te = np.array([0.4, 0.4, 0.1, 0.1])):

    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert np.abs(np.sum(tr_re_va_te) - 1) < 1e-6
 
    x_normed, x_normalizer = normalize(x)
    x = x_normed

    N_train, N_recal, N_val, N_test = (len(x) * tr_re_va_te).astype(int)

    idx_permu = np.random.choice(len(x), len(x), replace=False)

    train_idx = idx_permu[:N_train]
    recal_idx = idx_permu[N_train: N_train + N_recal]
    val_idx = idx_permu[N_train + N_recal: N_train + N_recal + N_val]
    test_idx = idx_permu[N_train + N_recal + N_val: ]

    
    
    train_X, recal_X, val_X, test_X = x[train_idx], x[recal_idx], x[val_idx], x[test_idx]

    train_Y, recal_Y, val_Y, test_Y = y[train_idx], y[recal_idx], y[val_idx], y[test_idx]

    return (train_X, train_Y), (recal_X, recal_Y), (val_X, val_Y), (test_X, test_Y)






def ts_data_formulator(x, y, window_size = 5):
    
    assert len(x) == len(y)
    
    reshaped_x = []
    
    for i in range(window_size-1, len(y)):
        
        reshaped_x.append(x[i-window_size+1:i+1])
        
    return np.array(reshaped_x), y[window_size-1:]




def california_housing_process(path = "Dataset/CaliforniaHousing/housing.csv"):

    df = pd.read_csv(path)
    df = df.dropna(axis = 0)

    # log transformation 
    t = 9e-1
    df['total_rooms'] = np.log(df['total_rooms'] + t)
    df['total_bedrooms'] = np.log(df['total_bedrooms'] + t)
    df['population']  = np.log(df['population'] +t)
    df['households'] = np.log(df['households'] + t)
    df['total_rooms'] = np.log(df['total_rooms'] + t)

    for column in df.drop(columns=['ocean_proximity','median_house_value' ]).columns:
        df[column] = (df[column] - np.mean(df[column])) / np.std(df[column])
        
    df = pd.get_dummies(df)

    x = np.array(df.drop(columns = ['median_house_value']).values)
    y = np.array(df.median_house_value.values) / 1E4

    return x, y



def OnlineNews(path = "Dataset/OnlineNewsPopular/OnlineNewsPopularity.csv"):

    # refer to https://www.kaggle.com/code/thehapyone/exploratory-analysis-for-online-news-popularity


    data = pd.read_csv(path)
    data.drop(labels=['url', ' timedelta'], axis = 1, inplace=True)


    data = data[data[' shares'] <= 10000]


    # Comment - Visualizing the n_non_stop_words data field shows that the present of a record with 1042 value, 
    # futher observation of that data shows that it belongs to entertainment which is not actually. It belongs to world news or others.
    # this particluar also contains 0 on a lot of attributes. This record is classifed as a noise and will be remove.
    data = data[data[' n_non_stop_words'] != 1042]
    # Here, we will go ahead and drop the field of ' n_non_stop_words'
    data.drop(labels=[' n_non_stop_words'], axis = 1, inplace=True)

    # remove noise from n_tokens_content. those equals to 0
    data  = data[data[' n_tokens_content'] != 0]

    x = np.array(data.drop(columns = [' shares']).values, dtype = float)

    y = np.array(data[' shares'].values, dtype = float) / 100

    return x, y

