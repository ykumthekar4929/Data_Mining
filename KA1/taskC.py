import data_handler
import centroid_classifier
import matplotlib.pyplot as plt

all_accs = []
# (train=5 test=34)
trainX, trainY, testX, testY = data_handler.splitData2TestTrain(\
                                                data_handler.pickDataClass(\
                                                   'Handwrittenletters.txt',
                                                    data_handler.letter_2_digit_convert(\
                                                               "adgjwryozm")),
                                                    39,
                                                    "1:34")


all_accs.append(centroid_classifier.predict(trainX, trainY, testX, testY, 4))
# (train=10 test=29),
trainX, trainY, testX, testY = data_handler.splitData2TestTrain(\
                                                data_handler.pickDataClass(\
                                                   'Handwrittenletters.txt',
                                                    data_handler.letter_2_digit_convert(\
                                                               "adgjwryozm")),
                                                    39,
                                                    "1:29")
all_accs.append(centroid_classifier.predict(trainX, trainY, testX, testY, 4))

# (train=15 test=24) ,
trainX, trainY, testX, testY = data_handler.splitData2TestTrain(\
                                                data_handler.pickDataClass(\
                                                   'Handwrittenletters.txt',
                                                    data_handler.letter_2_digit_convert(\
                                                               "adgjwryozm")),
                                                    39,
                                                    "1:24")
all_accs.append(centroid_classifier.predict(trainX, trainY, testX, testY, 4))

# (train=20 test=19)
trainX, trainY, testX, testY = data_handler.splitData2TestTrain(\
                                                data_handler.pickDataClass(\
                                                   'Handwrittenletters.txt',
                                                    data_handler.letter_2_digit_convert(\
                                                               "adgjwryozm")),
                                                    39,
                                                    "1:19")
all_accs.append(centroid_classifier.predict(trainX, trainY, testX, testY, 4))

# (train=25 test=24)
trainX, trainY, testX, testY = data_handler.splitData2TestTrain(\
                                                data_handler.pickDataClass(\
                                                   'Handwrittenletters.txt',
                                                    data_handler.letter_2_digit_convert(\
                                                               "adgjwryozm")),
                                                    39,
                                                    "1:14")
all_accs.append(centroid_classifier.predict(trainX, trainY, testX, testY, 4))

# (train=30 test=9)
trainX, trainY, testX, testY = data_handler.splitData2TestTrain(\
                                                data_handler.pickDataClass(\
                                                   'Handwrittenletters.txt',
                                                    data_handler.letter_2_digit_convert(\
                                                               "adgjwryozm")),
                                                    39,
                                                    "1:9")
all_accs.append(centroid_classifier.predict(trainX, trainY, testX, testY, 4))

# (train=35 test=4)
trainX, trainY, testX, testY = data_handler.splitData2TestTrain(\
                                                data_handler.pickDataClass(\
                                                   'Handwrittenletters.txt',
                                                    data_handler.letter_2_digit_convert(\
                                                               "adgjwryozm")),
                                                    39,
                                                    "1:4")
all_accs.append(centroid_classifier.predict(trainX, trainY, testX, testY, 4))


print all_accs

plt.plot(all_accs)
plt.show()
