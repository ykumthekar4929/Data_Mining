import data_handler
import LinearRegression
import centroid_classifier
import kn_classifier
import SVM
import numpy as np
import random

def fxcv_driver(k, train_data, feature_names):
    for index, item in enumerate(train_data):
        item.append(feature_names[index])

    random.shuffle(train_data)
    k_splits = np.array_split(train_data, k)
    feature_splits = [[in_item[-1] for in_item in item]for item in k_splits]
    all_accuracy =  0
    for k in range(0,k):
        print (k)
        trainX = []
        trainY = []
        testX = k_splits[k]
        testY = feature_splits[k]
        trainX_temp = k_splits[:k] + k_splits[(k + 1):]
        trainY_temp = feature_splits[:k] + feature_splits[(k + 1):]
        for x in range(len(trainX_temp)):
            trainX.extend(trainX_temp[x])
            trainY.extend(trainY_temp[x])
        # accuracy = (centroid_classifier.predict(trainX, trainY, testX, testY, 4))
        # accuracy = (SVM.get_score(trainX, trainY, testX, testY))
        # accuracy = (kn_classifier.knn_driver(trainX,  testX, 4))
        # print (accuracy)
        # import ipdb; ipdb.set_trace()

        all_accuracy += accuracy
    k_accuracy = float(all_accuracy)/5
    return k_accuracy

# data, indexes = data_handler.get_data("ATNTFaceImages400.txt")
# print (fxcv_driver(5, data, indexes))
trainX, trainY, testX, testY = data_handler.splitData2TestTrain('ATNTFaceImages400.txt', 10, '1:10')


print (SVM.cross_validate(trainX, trainY, testX, testY))
