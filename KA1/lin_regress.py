import numpy as np
import data_handler

# % The following is a Python code for Linear Regresion

trainX, trainY, testX, testY = data_handler.splitData2TestTrain("Handwrittenletters.txt", 30,  "1:20")

Ytrain = np.array(trainY, np.int32)
Xtrain = np.array(trainX, np.int32)
Xtest = np.array(testX, np.int32)
Ytest = np.array(testY, np.int32)

def linear(Xtrain, Xtest, Ytrain, Ytest):
	A_train = np.ones((1,len(trainX)))    # N_train : number of training instance
	A_test = np.ones((1,len(testX)))      # N_test  : number of test instance
	# Xtrain_padding = np.row_stack((Xtrain,A_train))
	# Xtest_padding = np.row_stack((Xtest,A_test))
	# import ipdb; ipdb.set_trace()

	B_padding = np.dot(np.linalg.pinv(Xtrain), Ytrain.T)   # (XX')^{-1} X  * Y'  #Ytrain : indicator matrix
	Ytest_padding = np.dot(B_padding.T,Xtest_padding)
	Ytest_padding_argmax = np.argmax(Ytest_padding,axis=0)+1
	err_test_padding = Ytest - Ytest_padding_argmax
	TestingAccuracy_padding = (1-np.nonzero(err_test_padding)[0].size/len(err_test_padding))*100

linear(Xtrain, Xtest, Ytrain, Ytest)

#
# A_train = np.ones((len(trainX),len(trainX[0])))    # N_train : number of training instance
# A_test = np.ones((len(testX),len(testX[0])))      # N_test  : number of test instance
#
# Xtrain_padding = np.row_stack((Xtrain,A_train))
# Xtest_padding = np.row_stack((Xtest,A_test))
#
# '''computing the regression coefficients'''
#
# B_padding = np.dot(np.linalg.pinv(Xtrain), Ytrain.T)   # (XX')^{-1} X  * Y'  #Ytrain : indicator matrix
# # import ipdb; ipdb.set_trace()
#
# Ytest_padding = np.dot(B_padding.T,Xtest.T)
# Ytest_padding_argmax = np.argmax(Ytest,axis=0)+1
# err_test_padding = Ytest - Ytest_padding_argmax
# TestingAccuracy_padding = (1-np.nonzero(err_test_padding)[0].size/len(err_test_padding))*100
# import ipdb; ipdb.set_trace()
#
# print TestingAccuracy_padding
