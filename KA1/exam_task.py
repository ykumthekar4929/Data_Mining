import data_handler
import LinearRegression
import centroid_classifier
import kn_classifier
import SVM

dataset_file = "Handwrittenletters.txt"
test_string = "DCIAYNKR" #4391 YVKR
test_instances = "1:10"
instances_pre_class = 39

trainX, trainY, testX, testY = data_handler.splitData2TestTrain(\
                                                data_handler.pickDataClass(\
                                                   dataset_file,
                                                    data_handler.letter_2_digit_convert(\
                                                       test_string)),
                                                    instances_pre_class,
                                                    test_instances)

data_handler.store_data(trainX, trainY, testX, testY)

print ("Actual classes %s"%data_handler.letter_2_digit_convert(test_string))
print ("Centroid method")
centroid_acc, prediction = centroid_classifier.predict(trainX, trainY, testX, testY, 3)
print ("Centroid accuracy %s"%centroid_acc)
print ("Predicted Classes %s"%prediction)
print ("============================================================================")
print ("KNN method")
kn_acc, prediction = kn_classifier.knn_driver(trainX, testX, 3)
print ("KNN accuracy %s"%kn_acc)
print ("Predicted Classes %s"%prediction)
print ("============================================================================")
print ("SVM method")
svm_acc, prediction = SVM.predict(trainX, trainY, testX, testY)
print ("Predicted Classes %s"%prediction)
print ("============================================================================")
print ("Linear Regression method")
result, lin_reg_acc = LinearRegression.predict(trainX, trainY, testX, testY)
print ("Linear Regression accuracy %s"%abs(lin_reg_acc))
print ("Predicted Classes %s"%result)
