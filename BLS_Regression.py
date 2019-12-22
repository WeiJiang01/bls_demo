# -*- coding: utf-8 -*-
"""
Revamp based on @HAN_RUIZHI on Wed. Dec 18 20:05 2019

@author: Wei Jiang

This code is the new version of BLS in Python. 

Campare to the previous version, this version divide the bls_regression into 2 part,
bls_regression_train and bls_regression_test. The training part return the prediction 
value and parameters.

This code change the activation function, which is insteaded by ReLU

You can save the parameters with pickle package, and test with function bls_regression_test.

"""
import numpy as np
from sklearn import preprocessing
from numpy import random
import time

def ReLU(x):
    return (np.abs(x) + x) / 2.0

def pinv(A,reg):
    return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)


'''
参数压缩
'''
def shrinkage(a,b):
    z = np.maximum(a - b, 0) - np.maximum( -a - b, 0)
    return z
'''
参数稀疏化
'''
def sparse_bls(A,b):
    lam = 0.001
    itrs = 50
    AA = np.dot(A.T,A)   
    m = A.shape[1]
    n = b.shape[1]
    wk = np.zeros([m,n],dtype = 'double')
    ok = np.zeros([m,n],dtype = 'double')
    uk = np.zeros([m,n],dtype = 'double')
    L1 = np.mat(AA + np.eye(m)).I
    L2 = np.dot(np.dot(L1,A.T),b)
    for i in range(itrs):
        tempc = ok - uk
        ck =  L2 + np.dot(L1,tempc)
        ok = shrinkage(ck + uk, lam)
        uk += ck - ok
        wk = ok
    return wk

def bls_regression_train(train_x,train_y,C,NumFea,NumWin,NumEnhan):
   
    u = 0
    WF = list()
    for i in range(NumWin):
        random.seed(i+u)
        WeightFea=2*random.randn(train_x.shape[1]+1,NumFea)-1
        WF.append(WeightFea)
    #    random.seed(100)
    WeightEnhan=2*random.randn(NumWin*NumFea+1, NumEnhan)-1
    time_start = time.time()
    H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0],1])])
    y = np.zeros([train_x.shape[0],NumWin*NumFea])
    WFSparse = list()
    distOfMaxAndMin = np.zeros(NumWin)
    meanOfEachWindow = np.zeros(NumWin)

    # Windows loop
    for i in range(NumWin):
        WeightFea = WF[i]
        A1 = H1.dot(WeightFea)        
        scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
        A1 = scaler1.transform(A1)
        WeightFeaSparse  = sparse_bls(A1,H1).T
        WFSparse.append(WeightFeaSparse)
    
        T1 = H1.dot(WeightFeaSparse)
        meanOfEachWindow[i] = T1.mean()
        distOfMaxAndMin[i] = T1.max() - T1.min()
        T1 = (T1 - meanOfEachWindow[i])/distOfMaxAndMin[i] 
        y[:,NumFea*i:NumFea*(i+1)] = T1

    H2 = np.hstack([y,0.1 * np.ones([y.shape[0],1])])
    T2 = H2.dot(WeightEnhan)
    T2 = ReLU(T2)
    T3 = np.hstack([y,T2])
    WeightTop = pinv(T3,C).dot(train_y)

    Training_time = time.time()- time_start
    print('Training has been finished!')
    print('The Total Training Time is : ', round(Training_time,6), ' seconds' )
    NetoutTrain = T3.dot(WeightTop)

    time_start = time.time()
    return NetoutTrain, NumWin,  WeightEnhan, WFSparse,WeightTop,meanOfEachWindow,distOfMaxAndMin
    
def bls_regression_test(test_x,NumFea,NumWin,WeightEnhan, WFSparse, WeightTop ,meanOfEachWindow,distOfMaxAndMin):
    HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0],1])])
    yy1=np.zeros([test_x.shape[0],NumWin*NumFea])
    for i in range(NumWin):
        WeightFeaSparse = WFSparse[i]
        TT1 = HH1.dot(WeightFeaSparse)
        TT1  = (TT1 - meanOfEachWindow[i])/distOfMaxAndMin[i]   
        yy1[:,NumFea*i:NumFea*(i+1)] = TT1

    HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0],1])])
    TT2 = ReLU(HH2.dot( WeightEnhan))
    TT3 = np.hstack([yy1,TT2])
    NetoutTest = TT3.dot(WeightTop)

    return NetoutTest

