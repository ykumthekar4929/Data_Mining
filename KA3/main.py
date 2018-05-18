from operator import itemgetter
import numpy as np
import pandas as pd
import math
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
import pprint

data_frame = pd.read_csv('GenomeTrainXY.txt', sep=",", header=None)
test_data_frame = pd.read_csv('GenomeTestX.txt', sep=",", header=None)

data_frame = data_frame.T
# data_frame = data_frame
test_data_frame = test_data_frame.T
# test_data_frame = test_data_frame

data_frame = data_frame.astype('float')
test_data_frame = test_data_frame.astype('float')

class_labels = data_frame.iloc[:, 0]
# class_labels = data_frame.iloc[:, -1]  #if class labels are at end
# data = data_frame.iloc[:, 1:]
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

def classify_and_predict(clf, train_x, train_y, test_x, method):
        print ("Method %s:\n"%method)
        clf.fit(train_x, train_y)
        print(clf.predict(test_x))

def driver():

        # for i in range(0, data_frame.shape[1] - 1):
        for i in range(1, data_frame.shape[1]):
                compute_f_test(data_frame.iloc[:, [0, i]], i)

        f_data_des = np.array(sorted(f_data, key=itemgetter(1), reverse=True))

        top_selected_samples = f_data_des[:100, :]

        top_100_ixs = [item[0] for item in top_selected_samples]
        top_100_test_ixs = [i-1 for i in top_100_ixs]




        print ("Selected top %s sample indexes and f_scores \n"%len(top_selected_samples))
        for index, item in enumerate(top_selected_samples):
                print ("%s => %s - %s"%(index+1, item[0], item[1]))


        test_data = test_data_frame.iloc[:, top_100_test_ixs]
        top_100_ixs = np.insert(top_100_ixs, 0, 0)
        training_data  = data_frame.iloc[:, top_100_ixs]
        train_x = training_data.iloc[:,1:]
        train_y = training_data.iloc[:,0]
        train_x = train_x.as_matrix()
        train_y = train_y.as_matrix()
        train_y = train_y.astype('int')
        # pprint.pprint(f_data_des[0:100])
        # pprint.pprint(top_100_ixs)


        print ("\nPredicted Classes : \n")
        classify_and_predict(linear_model.LinearRegression(), train_x, train_y, test_data, "Linear Regression")
        classify_and_predict(SVC(kernel='linear'), train_x, train_y, test_data, "Support Vector Machine")
        classify_and_predict(KNeighborsClassifier() , train_x, train_y, test_data, "KNN Classifier")
        classify_and_predict(NearestCentroid() , train_x, train_y, test_data, "Centroid Method")

driver()
