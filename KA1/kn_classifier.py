import math
import sys
import operator
import random

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


def knn_driver(train_data, test_data, k):
	test_results = [getNeighbors(train_data, testInstance, k) for testInstance in test_data]
	test_metrics = [[item[-1] for item in neighbor] for neighbor in test_results]

	test_metrics_result = [mode(item) for item in test_metrics]

	test_pos_count = 0
	for index, item in enumerate(test_data):
		if item[-1] == mode(test_metrics[index]):
			test_pos_count+=1
	return (((test_pos_count/len(test_metrics))*100)),test_metrics_result

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
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = getEuclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def main():
	dataset = get_data()
	train_data, test_data = train_test_split(dataset)
	for k in range(1,11):
		knn_driver(train_data, test_data, k)



if __name__ == '__main__':
	# print(sys.argv)
	main()
