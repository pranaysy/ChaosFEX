# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:40:16 2020

@author: harik
"""
import numpy as np
from chaos_reg_func import chaos_train, chaos_prediction
from sklearn.metrics import mean_squared_error
from load_data import get_data
import matplotlib.pyplot as plt
import numpy as np
import os

# Axes3D import has side effects, it enables using projection='3d' in add_subplot


LENGTH = 10000
EPSILON = 0.001
THRESHOLD = 0.499
DIRECTION_ITER = "backward"
K1 = np.arange(1,50,5)
seed = 400 
np.random.seed(seed)
RANDOM_SEQUENCE = [np.random.randint(0, 2) for t in range(0, LENGTH)]

MSE = []
                   
DATA_NAME = 'multi_variate_data'
X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = get_data(DATA_NAME)


FIRINGTIME = chaos_train(X_TRAIN, Y_TRAIN,THRESHOLD, LENGTH, DIRECTION_ITER, RANDOM_SEQUENCE, EPSILON, DATA_NAME)

for K in K1:
    
    Y_PRED = chaos_prediction(X_TEST, X_TRAIN, Y_TRAIN, FIRINGTIME, K, DIRECTION_ITER, THRESHOLD, RANDOM_SEQUENCE, DATA_NAME)
    MSE.append(mean_squared_error(Y_PRED, Y_TEST))
    
    if np.max([X_TRAIN.shape[1], Y_TRAIN.shape[1]]) == 1:
        ### The figure is generated for 2-dimensional case
        print("Creating Results Path")
        # define the name of the directory to be created
        path = os.getcwd()
        resultpath = path + '/chaos-regression-results/'  + DATA_NAME + '/images/'
    
        # define the name of the directory to be created


        try:
            os.makedirs(resultpath)
        except OSError:
            print ("Creation of the result directory %s failed" % resultpath)
        else:
            print ("Successfully created the result directory %s" % resultpath)
        
        
        plt.figure(figsize=(10,10))
        plt.plot(X_TEST,Y_TEST,'ok', markersize = 4, label = 'actual value')
        plt.plot(X_TEST,Y_PRED,'*r', markersize = 7, label = 'predicted value')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True)
        plt.xlabel('X', fontsize=20)
        plt.ylabel('Prediction', fontsize=20)
        plt.legend(bbox_to_anchor=(0.38,0.94),fontsize = 17)
        plt.savefig(resultpath +'/testdata_pred_' + str(K) + '_.jpg', format='jpg', dpi=700)
        plt.show()
        
    elif (X_TRAIN.shape[1] == 2) and (Y_TRAIN.shape[1] == 1):
        
        print("Creating Results Path")
        # define the name of the directory to be created
        path = os.getcwd()
        resultpath = path + '/chaos-regression-results/'  + DATA_NAME + '/images/'
    
        # define the name of the directory to be created


        try:
            os.makedirs(resultpath)
        except OSError:
            print ("Creation of the result directory %s failed" % resultpath)
        else:
            print ("Successfully created the result directory %s" % resultpath)
        from mpl_toolkits.mplot3d import Axes3D  
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(X_TEST[:,0], X_TEST[:,1], Y_TEST[:,0], '*b', label = 'actual value')
        ax.plot(X_TEST[:,0], X_TEST[:,1], Y_PRED[:,0], '*r', label = 'predicted value')
        ax.set_xlabel('X', fontsize = 18)
        ax.set_ylabel('Y ', fontsize = 18)
        ax.set_zlabel('f(X,Y)', fontsize = 18)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(17)
        plt.legend(bbox_to_anchor=(0.38,0.94),fontsize = 17)
        plt.savefig( resultpath +'/testdata_pred_' + str(K) + '_.jpg', format='jpg', dpi=700)
        plt.show()

filepath = os.getcwd()
resultpath = filepath + '/chaos-regression-results/' + DATA_NAME       
np.save(resultpath +'/MSE', MSE) 
np.save(resultpath +'/neighbours', K1)