import data_handler
import LinearRegression
import centroid_classifier
import kn_classifier
import SVM

trainX, trainY, testX, testY = data_handler.splitData2TestTrain(\
                                                data_handler.pickDataClass(\
                                                   'Handwrittenletters.txt',
                                                    data_handler.letter_2_digit_convert(\
                                                               "abcde")),
                                                    39,
                                                    "1:9")


print ("Centroid method")
centroid_acc = centroid_classifier.predict(trainX, trainY, testX, testY, 4)
print ("Centroid accuracy %s"%centroid_acc)


print ("KNN method")
kn_acc = kn_classifier.knn_driver(trainX, testX, 4)
print ("KNN accuracy %s"%kn_acc)



print ("SVM method")
svm_acc = SVM.predict(trainX, trainY, testX, testY)
# print ("KNN accuracy %s"%svm_acc)



print ("Linear Regression method")
result, lin_reg_acc = LinearRegression.predict(trainX, trainY, testX, testY)
print ("Linear Regression accuracy %s"%lin_reg_acc)
