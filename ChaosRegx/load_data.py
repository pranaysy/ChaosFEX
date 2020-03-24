# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:11:57 2020

@author: harik
"""


import logging
import numpy as np
import pandas as pd
def get_data(DATA_NAME):
    """
    

    Parameters
    ----------
    DATA_NAME : string
        Data folder name

    Returns
    -------
    Normalized train data and test data in the range open interval -> (0,1)

    """
    
#     if DATA_NAME == "multi_variate_data" or "linear_data":
    folder_path = "data/" + DATA_NAME + "/" 


    X_train = np.array( pd.read_csv(folder_path+"X_train.csv", header = None) )
    #trainlabel =  np.array( pd.read_csv(folder_path+"y_train.csv", header = None) )
    y_train =  np.array( pd.read_csv(folder_path+"y_train.csv", header = None) )

    X_test = np.array( pd.read_csv(folder_path+"X_test.csv", header = None) )
    y_test = np.array( pd.read_csv(folder_path+"y_test.csv", header = None) )



    ## Data_normalization - A Compulsory step

    print(" ----------Step -1---------------")
    print("                                 ")
    print(" Data normalization done ")
    nMax_Xtrain = np.max(X_train) + 10**-8
    nMin_Xtrain = np.min(X_train) - 10**-8
    nMax_Xtest = np.max(X_test) + 10**-8
    nMin_Xtest = np.min(X_test) -10**-8
    X_train_norm = (X_train - nMin_Xtrain)/np.float(nMax_Xtrain - nMin_Xtrain)
    X_test_norm = (X_test - nMin_Xtest)/np.float(nMax_Xtest - nMin_Xtest)



    nMax_ytrain = np.max(y_train) + 10**-8
    nMin_ytrain = np.min(y_train) - 10**-8
    nMax_ytest = np.max(y_test) + 10**-8
    nMin_ytest = np.min(y_test) -10**-8
    y_train_norm = (y_train - nMin_ytrain)/np.float(nMax_ytrain - nMin_ytrain)
    y_test_norm = (y_test - nMin_ytest)/np.float(nMax_ytest - nMin_ytest)

    try:
        assert np.min(X_train_norm) >= 0.0 and np.max(X_train_norm <= 1.0)
    except AssertionError:
        return logging.error("Train Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)

    try:
        assert np.min(X_test_norm) >= 0.0 and np.max(X_test_norm <= 1.0)
    except AssertionError:
        return logging.error("Test Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)

    return X_train_norm, y_train_norm, X_test_norm, y_test_norm


def get_data_signal_denoise(DATA_NAME):
    """
    

    Parameters
    ----------
    DATA_NAME : string
        Data folder name

    Returns
    -------
    Normalized train data and test data in the range open interval -> (0,1)

    """
    
#     if DATA_NAME == "multi_variate_data" or "linear_data":
    folder_path = "data/" + DATA_NAME + "/" 


    X_train = np.array( pd.read_csv(folder_path+"X_value.csv", header = None) )
    #trainlabel =  np.array( pd.read_csv(folder_path+"y_train.csv", header = None) )
    y_train =  np.array( pd.read_csv(folder_path+"y_value_noise.csv", header = None) )

    X_test = np.array( pd.read_csv(folder_path+"X_value.csv", header = None) )
    y_test = np.array( pd.read_csv(folder_path+"y_value.csv", header = None) )



    ## Data_normalization - A Compulsory step

    print(" ----------Step -1---------------")
    print("                                 ")
    print(" Data normalization done ")
    nMax_Xtrain = np.max(X_train) + 10**-8
    nMin_Xtrain = np.min(X_train) - 10**-8
    nMax_Xtest = np.max(X_test) + 10**-8
    nMin_Xtest = np.min(X_test) -10**-8
    X_train_norm = (X_train - nMin_Xtrain)/np.float(nMax_Xtrain - nMin_Xtrain)
    X_test_norm = (X_test - nMin_Xtest)/np.float(nMax_Xtest - nMin_Xtest)



    nMax_ytrain = np.max(y_train) + 10**-8
    nMin_ytrain = np.min(y_train) - 10**-8
    nMax_ytest = np.max(y_test) + 10**-8
    nMin_ytest = np.min(y_test) -10**-8
    y_train_norm = (y_train - nMin_ytrain)/np.float(nMax_ytrain - nMin_ytrain)
    y_test_norm = (y_test - nMin_ytest)/np.float(nMax_ytest - nMin_ytest)

    try:
        assert np.min(X_train_norm) >= 0.0 and np.max(X_train_norm <= 1.0)
    except AssertionError:
        return logging.error("Train Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)

    try:
        assert np.min(X_test_norm) >= 0.0 and np.max(X_test_norm <= 1.0)
    except AssertionError:
        return logging.error("Test Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)

    return X_train_norm, y_train_norm, X_test_norm, y_test_norm
