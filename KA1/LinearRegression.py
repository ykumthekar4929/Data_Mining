# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:38:18 2018

@author: pradn
"""

# from sklearn.model_selection import cross_val_score
import numpy as np
import data_handler
import matplotlib.pyplot as plt

#trainX, trainY, testX, testY = data_handler.splitData2TestTrain("ATNTFaceImages400.txt", 10, '1:7')
# trainX, trainY, testX, testY = data_handler.splitData2TestTrain(data_handler.pickDataClass('HandWrittenLetters.txt',data_handler.letter_2_digit_convert("ABCD")), 39, '1:9')



def predict(trainX, trainY, testX, testY):
    Xtrain=np.array(trainX,np.int32)
    Ytrain=np.array(trainY,np.int32)
    Xtest=np.array(testX,np.int32)
    Ytest=np.array(testY,np.int32)
    A_train = np.ones((len(trainX),len(trainX[0])))
    A_test = np.ones((len(testX),len(testX[0])))
    Xtrain_padding = np.row_stack((Xtrain,A_train))
    Xtest_padding = np.row_stack((Xtest,A_test))
    '''computing the regression coefficients'''
    B_padding = np.dot(np.linalg.pinv(Xtrain), Ytrain.T)
    Ytest_padding = np.dot(B_padding.T,Xtest.T)
    Ytest_padding_argmax = np.argmax(Ytest_padding,axis=0)+1
    err_test_padding = Ytest - Ytest_padding_argmax
    TestingAccuracy_padding = -(float((1-np.nonzero(err_test_padding)[0].size)/len(err_test_padding)))*100
    return Ytest_padding, TestingAccuracy_padding

# print (predict(trainX, trainY, testX, testY ))
