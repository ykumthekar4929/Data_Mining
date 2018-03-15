# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 18:17:33 2018

@author: pradn
"""
import numpy as np
from sklearn import svm
import data_handler
from sklearn import preprocessing
from sklearn import metrics

from sklearn.model_selection import cross_val_score

le = preprocessing.LabelEncoder

#trainX, trainY, testX, testY = data_handler.splitData2TestTrain("ATNTFaceImages400.txt", 10, '1:6')
#trainX = [item[:-1] for item in trainX]

def get_score(trainX, trainY, testX, testY):
    return predict(trainX, trainY, testX, testY).score

def predict(trainX, trainY, testX, testY):
    Xtrain=np.array(trainX,np.int32)
    Ytrain=np.array(trainY,np.int32)
    Xtest=np.array(testX,np.int32)
    Ytest=np.array(testY,np.int32)
    clf=svm.SVC()
    clf.fit(Xtrain,Ytrain)
    predicted= clf.predict(testX)
    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(Ytest, predicted)))

    return clf
    # print("Confusion matrix:\n%s" % metrics.confusion_matrix(Ytest, predicted))
    #plt.scatter(Xtrain[:,0].reshape(Xtrain[:,1].shape),Xtrain[:,1])
    #plt.plot(Ytest,result

def cross_validate(trainX, trainY, testX, testY):
    clf = predict(trainX, trainY, testX, testY)
    scores = cross_val_score(clf, Xtest, Ytest, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()
    # print("Scores %s" %(scores))

# trainX, trainY, testX, testY = data_handler.splitData2TestTrain(data_handler.pickDataClass('HandWrittenLetters.txt',data_handler.letter_2_digit_convert("ABCDE")), 39, '1:30')
# predict(trainX, trainY, testX, testY)
