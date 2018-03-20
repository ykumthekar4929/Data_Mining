import data_handler
import centroid_classifier
import matplotlib.pyplot as plt
all_accs = []

sample_string = "adgjwryozm"
splits = ["1:34", "1:29", "1:24", "1:19", "1:14", "1:9", "1:4"]
for split in splits:
    print ("Current split %s"%split)
    trainX, trainY, testX, testY = data_handler.splitData2TestTrain(\
                                                    data_handler.pickDataClass(\
                                                       'Handwrittenletters.txt',
                                                        data_handler.letter_2_digit_convert(\
                                                            sample_string       )),
                                                        39,
                                                        split)


    all_accs.append(centroid_classifier.predict(trainX, trainY, testX, testY, 4))

print all_accs

plt.plot(all_accs)
plt.show()
