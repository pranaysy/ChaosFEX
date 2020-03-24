# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:14:20 2020

@author: harik
"""
from skew_tent import skew_tent, iterations
import numpy as np
import numpy as np
#from numba import vectorize, float64, njit
import os
import time

def chaos_train(X_TRAIN, Y_TRAIN,THRESHOLD, LENGTH, DIRECTION_ITER, RANDOM_SEQUENCE, EPSILON, DATA_NAME):
    """
    
    Parameters
    ----------
    X_TRAIN : 2d array, float64
        Normalized matrix in the range (0,1)
    Y_TRAIN : 2d array, float64
        Normalized matrix in the range (0,1)
    THRESHOLD : scalar, float64
        Threshold value for the skew-tent map.
    LENGTH : scalar, integer
        Length of the chaotic TRAJECTORY.
    DIRECTION_ITER : string
        DIRECTION_ITER == forward computes forward iteration,
        DIRECTION_ITER == backward computes backward iteration
    RANDOM_SEQUENCE : array, integer 
        A randomly generated array of ones and zeros
    EPSILON : scalar, float64
        a value between 0 and 0.3
    DATA_NAME : string
        the name of the data folder

    Returns
    -------
    FIRINGTIME : 2d numpy array
        The time required to reach the neighbourhood of stimulus.
        Reference: Harikrishnan, N. B., and Nithin Nagaraj. 
        "A novel chaos theory inspired neuronal architecture." 2019 Global Conference for Advancement in Technology (GCAT). IEEE, 2019.

    """
    
    
    start_time = time.time()
    
    #####Creating Results Folder
    
    print("Creating Results Path")
    # define the name of the directory to be created
    path = os.getcwd()
    resultpath = path + '/chaos-regression-results/'  + DATA_NAME 

    # define the name of the directory to be created


    try:
        os.makedirs(resultpath)
    except OSError:
        print ("Creation of the result directory %s failed" % resultpath)
    else:
        print ("Successfully created the result directory %s" % resultpath)
    
    
    NUM_INSTANCE_X_TRAIN = X_TRAIN.shape[0]
    NUM_FEATURES_X_TRAIN = X_TRAIN.shape[1]
    NUM_FEATURES_Y_TRAIN = Y_TRAIN.shape[1]

    FIRINGTIME = np.zeros((NUM_INSTANCE_X_TRAIN, NUM_FEATURES_X_TRAIN * NUM_FEATURES_Y_TRAIN))
    for TRAIN_ROW in range(0, NUM_INSTANCE_X_TRAIN):
        COL_VAL = 0
        SHIFT = 0
        for TRAIN_COL in range(0, NUM_FEATURES_X_TRAIN):

            TRAJECTORY = iterations(X_TRAIN[TRAIN_ROW,TRAIN_COL], THRESHOLD, LENGTH, DIRECTION_ITER, RANDOM_SEQUENCE)

            FIRING_TIME_LIST = []

            for PRED_COL in range(0, NUM_FEATURES_Y_TRAIN):
                
                if (np.abs(Y_TRAIN[TRAIN_ROW,PRED_COL] - TRAJECTORY) < EPSILON).tolist().count([True]) == 0:
                       return  print("Initial condtion", Y_TRAIN[TRAIN_ROW,PRED_COL]," or discrimination threshold ",THRESHOLD, " does not satisfy TT property")
                        
                
                A = (np.abs((Y_TRAIN[TRAIN_ROW,PRED_COL]) - TRAJECTORY) < EPSILON).argmax()
                
                FIRING_TIME_LIST.append(A)

            FIRINGTIME[TRAIN_ROW,COL_VAL:Y_TRAIN.shape[1] + SHIFT] =  FIRING_TIME_LIST
            COL_VAL = COL_VAL + Y_TRAIN.shape[1]
            SHIFT = SHIFT+Y_TRAIN.shape[1]
    FIRINGTIME = FIRINGTIME.astype(np.int64)
       
    np.save(resultpath+"/firingtime_train", FIRINGTIME )
    np.save(resultpath+"/RANDOM_SEQUENCE", RANDOM_SEQUENCE )
    np.save(resultpath+"/B", [THRESHOLD] , )
    print("Time for training--- %s seconds ---" % (time.time() - start_time)) 
    return FIRINGTIME
       

def chaos_prediction(X_TEST, X_TRAIN, Y_TRAIN, FIRINGTIME, K, DIRECTION_ITER, THRESHOLD, RANDOM_SEQUENCE, DATA_NAME):
    """
    
    Parameters
    ----------
    X_TEST : 2d array, float64
        Normalized matrix in the range (0,1)
    X_TRAIN : 2d array, float64
        Normalized matrix in the range (0,1)
    Y_TRAIN : 2d array, float64
        Normalized matrix in the range (0,1)
    FIRINGTIME : 2d array, float64
        Normalized matrix in the range (0,1)
    K : scalar, integer
        K represents the number of nearest neighbours
    DIRECTION_ITER : string
        DIRECTION_ITER == forward computes forward iteration,
        DIRECTION_ITER == backward computes backward iteration
    RANDOM_SEQUENCE : array, integer 
        A randomly generated array of ones and zeros
    EPSILON : scalar, float64
        a value between 0 and 0.3
    DATA_NAME : string
        the name of the data folder

    Returns
    -------
    OUTPUT : 2d array, float64
        Prediction for Test data

    """
    
    start_time = time.time()
    
    
    print("Creating Results Path")
    # define the name of the directory to be created
    path = os.getcwd()
    resultpath = path + '/chaos-regression-results/'  + DATA_NAME 

    # define the name of the directory to be created


    # define the access rights
    #access_rights = 0o755

    try:
        os.makedirs(resultpath)
    except OSError:
        print ("Creation of the result directory %s failed" % resultpath)
    else:
        print ("Successfully created the result directory %s" % resultpath)

    NUM_INSTANCE_X_TEST = X_TEST.shape[0]
    NUM_FEATURES_X_TEST = X_TEST.shape[1]
    NUM_FEATURES_Y_TRAIN = Y_TRAIN.shape[1]
    NUM_FEATURES_X_TRAIN = X_TRAIN.shape[1]

    PREDICTION_MAT = np.zeros((K*NUM_INSTANCE_X_TEST, NUM_FEATURES_X_TEST*NUM_FEATURES_Y_TRAIN))
    FIRING_ROW = 0
    for TEST_ROW in range(0, NUM_INSTANCE_X_TEST):

        FIRING_COL = 0

        for TEST_COL in range(0, NUM_FEATURES_X_TEST):

            SORT_IND = np.argsort(np.abs(X_TEST[TEST_ROW, TEST_COL] - X_TRAIN[:,TEST_COL]))
            MAX_LENGTH = np.max([FIRINGTIME[SORT_IND[0 : K], FIRING_COL:FIRING_COL + NUM_FEATURES_Y_TRAIN].astype(np.int64)]) + 1
            APPROX_FIRINGTIME = FIRINGTIME[SORT_IND[0 : K], FIRING_COL:FIRING_COL + NUM_FEATURES_Y_TRAIN].astype(np.int64)
            TRAJECTORY_TEST = iterations(X_TEST[TEST_ROW,TEST_COL], THRESHOLD,MAX_LENGTH, DIRECTION_ITER, RANDOM_SEQUENCE)
            PREDICTION_MAT[FIRING_ROW:FIRING_ROW + K,  FIRING_COL:FIRING_COL + NUM_FEATURES_Y_TRAIN]= TRAJECTORY_TEST[APPROX_FIRINGTIME,0]
    #         PREDICTION_MAT[TEST_ROW,  FIRING_COL:FIRING_COL+NUM_FEATURES_Y_TRAIN]= np.mean(TRAJECTORY_TEST[APPROX_FIRINGTIME,0])
            FIRING_COL=  FIRING_COL + NUM_FEATURES_Y_TRAIN
  
        FIRING_ROW = FIRING_ROW + K
    
    OUTPUT = np.mean(np.split(np.mean(np.split(PREDICTION_MAT, NUM_FEATURES_X_TRAIN, axis = 1), axis = 0), NUM_INSTANCE_X_TEST, axis = 0),axis = 1)        
    np.save(resultpath+"/y_pred_" + str(K), OUTPUT )
    print("Time for prediction--- %s seconds ---" % (time.time() - start_time)) 
    return OUTPUT            
        


