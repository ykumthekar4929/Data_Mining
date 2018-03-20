import data_handler
import LinearRegression
import centroid_classifier
import kn_classifier
import SVM
import numpy as np
import random

def cross_validator(k, train_data, feature_names, classifier):
    for index, item in enumerate(train_data):
        item.append(feature_names[index])
    random.shuffle(train_data)
    k_splits = np.array_split(train_data, k)
    feature_splits = [[in_item[-1] for in_item in item]for item in k_splits]
    all_accuracy =  0
    for k in range(0,k):
        print ("For %s fold" %(int(k)+1))
        trainX = []
        trainY = []
        testX = k_splits[k]
        testY = feature_splits[k]
        trainX_temp = k_splits[:k] + k_splits[(k + 1):]
        trainY_temp = feature_splits[:k] + feature_splits[(k + 1):]
        for x in range(len(trainX_temp)):
            trainX.extend(trainX_temp[x])
            trainY.extend(trainY_temp[x])
        if classifier == 1:
            accuracy = (kn_classifier.knn_driver(trainX,  testX, 4))
        elif classifier == 2:
            accuracy = (centroid_classifier.predict(trainX, trainY, testX, testY, 4))
        elif classifier == 3:
            matrix, accuracy = (LinearRegression.predict(trainX, trainY, testX, testY))
        print (abs(accuracy))
        all_accuracy += accuracy
    k_accuracy = float(all_accuracy)/5
    return abs(k_accuracy)

def getTitle(classifier):
    return {
        '1':"KNN Classifier",
        '2':"Centroid Classifier",
        '3':"Linear regression",
        '4':"Support vector Machine"
    }[str(classifier)]

def driver(classifier):
    print (getTitle(classifier))
    if classifier == 4:
        trainX, trainY, testX, testY = data_handler.splitData2TestTrain('ATNTFaceImages400.txt', 10, '1:10')
        print ("\nAverage Accuracy for 5 folds: %s"% SVM.cross_validate(trainX, trainY, testX, testY))
    else:
        data, indexes = data_handler.get_data("ATNTFaceImages400.txt")
        print ("\nAverage Accuracy for 5 folds: %s"%cross_validator(5, data, indexes, classifier))


def main():
    for i in range(1,5):
        driver(i)

if __name__ == '__main__':
    main()
