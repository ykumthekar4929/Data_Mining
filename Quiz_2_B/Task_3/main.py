from operator import itemgetter
import numpy as np
import pandas as pd
import math
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
import pprint
import data_handler

data_frame = pd.read_csv('HandWrittenLetters.txt', sep=",", header=None)

data_frame = data_frame.T

data_frame = data_frame.astype('float')
class_labels = data_frame.iloc[:, 0]
f_data = []

def compute_f_test(f_vector, i):
        f_vector_mean = f_vector.iloc[:, -1].mean()
        f_vector_var  = f_vector.iloc[:, -1].var()
        per_class_vars = []
        per_class_means = []
        per_class_data = []
        unique_classes = f_vector.iloc[:, 0].unique()

        for index, label in enumerate(unique_classes):
                per_class_instances = f_vector.loc[f_vector.iloc[:, 0] == label]
                pci_mean = per_class_instances.iloc[:, 1].mean()
                pci_var = per_class_instances.iloc[:, 1].var()
                per_class_vars.append(pci_var)
                per_class_means.append(pci_mean)
                per_class_data.append([label, len(per_class_instances), pci_var, pci_mean])
        pooled_var = sum([ ((class_var[1] - 1) * class_var[2] )  for class_var in per_class_data])/(f_vector.shape[0] - len(unique_classes))
        f_score = 10000000000.00
        if pooled_var != 0.00:
                f_score_num_num = sum([ (c_data[1] * math.pow((c_data[3] - f_vector_mean), 2 ) ) for c_data in per_class_data])
                f_score_num = f_score_num_num / (len(unique_classes) -1)
                f_score = f_score_num / pooled_var

        f_data.append([i, round(f_score, 4), round(pooled_var, 4)])

def classify_and_predict(clf, train_x, train_y, test_x):
        clf.fit(train_x, train_y)
        results = clf.predict(test_x)
        print(results)

def driver():

        for i in range(1, data_frame.shape[1]):
                compute_f_test(data_frame.iloc[:, [0, i]], i)

        f_data_des = np.array(sorted(f_data, key=itemgetter(1), reverse=True))
        top_selected_samples = f_data_des[:200, :]
        top_100_ixs = top_selected_samples.astype('int')


        print ("Selected top %s sample indexes and f_scores \n"%len(top_selected_samples))
        for index, item in enumerate(top_selected_samples):
                print ("%s => %s - %s"%(index+1, item[0], item[1]))

        top_100_ixs = np.insert(top_100_ixs, 0, 0)
        training_data  = data_frame.iloc[:, top_100_ixs]
        train_x, train_y, testX, testY = data_handler.splitData2TestTrain(training_data.as_matrix().tolist(), 39,  "20:19")
        print ("\nPredicted Classes : \n")
        classify_and_predict(SVC(kernel='linear'), train_x, train_y, testX)
driver()
