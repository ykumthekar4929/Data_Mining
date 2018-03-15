import math
import sys
import operator
import random
import data_handler

train_data = []

def get_data():
	return pre_processor(file = open('ATNT50/trainDataXY.txt'))

def pre_processor(file):
	indexes = []
	data_raw = []
	data = []
	for index, line in enumerate(file):
		if index != 0:
			data_raw.append(line.rstrip().rsplit(','))
		else:
			indexes = line.rstrip().rsplit(',')
	for x in range(0, len(indexes)):
		index_list = [sample[x] for sample in data_raw]
		data.append(index_list + [indexes[x]])
	return data

def getCentroid(labelVectors):
	curr_centroid = []
	for j in range(len(labelVectors[0]) -1):
		curr_sum = 0
		for i in range(len(labelVectors)):
			curr_sum += int(labelVectors[i][j])
		curr_sum = float(curr_sum) / len(labelVectors)
		curr_centroid.append(curr_sum)
	curr_centroid.append(labelVectors[-1][-1])
	return curr_centroid
	# return [ float(sum([item[j] for item in labelVectors]))/len(labelVectors) for j in range(len(labelVectors[0]))]


def predict(trainX, trainY, testX, testY, k):

	unique_trainY = list(set(trainY))
	clustersDict = {}
	total_post_count = 0

	for x in range (len(trainY)):
		clustersDict.setdefault(trainY[x],[]).append(trainX[x])


	for testInstance in testX:
		train_data = [getCentroid(clustersDict[key]) for key in clustersDict.keys()]
		instanceResults = getNeighbors(train_data, testInstance, k)
		predictedClass = mode([item[-1] for item in instanceResults])
		if predictedClass == testInstance[-1]:
			total_post_count+=1
			clustersDict.setdefault(predictedClass).append(testInstance)

	# 	import ipdb; ipdb.set_trace()
		# print "Hello"
	# test_results = [getNeighbors(train_data, testInstance, k) for testInstance in testX]
	# predictions = [mode([item[-1] for item in result]) for result in test_results]
	#
	# for index, prediction in enumerate(predictions):
	# 	if prediction == testX[index][-1]:
	# 		total_post_count += 1

	return (float(total_post_count)/len(testX))*100.00
	# end = timer()
	# print end - start


def mode(numbers):
    largestCount = 0
    modes = []
    for x in numbers:
        if x in modes:
            continue
        count = numbers.count(x)
        if count > largestCount:
            del modes[:]
            modes.append(x)
            largestCount = count
        elif count == largestCount:
            modes.append(x)
    return modes[0]

def train_test_split(data):
	split_randInt = random.randint(int(len(data)/2), (len(data)-5))
	random.shuffle(data)
	train_data = data[0:split_randInt]
	test_data = data[split_randInt:-1]
	return train_data, test_data

def getEuclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((float(instance1[x]) - float(instance2[x])), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance) - 1
	for x in range(len(trainingSet)):
		dist = getEuclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def main(filename):
	# dataset, indexes = data_handler.get_data(filename)
	trainX, trainY, testX, testY = data_handler.splitData2TestTrain(data_handler.pickDataClass('Handwrittenletters.txt', data_handler.letter_2_digit_convert("ABCDEFGHIJ")), 39, "1:20")
	predict(trainX, trainY, testX, testY, 4)

if __name__ == '__main__':
	filename = sys.argv[1]
	main(filename)
