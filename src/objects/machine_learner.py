import threading
from .k_nearest_neighbor import KNearestNeighbor

class MachineLearner(threading.Thread):

    def __init__(self, client):
        super().__init__(daemon=True, target=self.run)
        self.__client = client
        self.__k_nearest_neighbor = None


    def run(self):
        print("Machine Learner initialized...")

    @property
    def k_nearest_neighbor(self):
        return self.__k_nearest_neighbor

    @property
    def client(self):
        return self.__client

    def initialize_k_nearest_neighbor(self, k):
        self.__k_nearest_neighbor = KNearestNeighbor(self, k)
        self.__k_nearest_neighbor.start()


    # # Load the data into the training and test sets from the dataset file
    # def load_dataset(self, split):
    #     with open('datasets/our_dataset.txt') as myfile:
    #         lines = myfile.readlines()
    #         dataset = []
    #
    #         # Create a 2d array with the numbers in the dataset file
    #         for line in lines:
    #             array = []
    #             for number in line.split():
    #                 array.append(float(number))
    #             dataset.append(array)
    #
    #         # Randomly add datasets to either test set or training set based on split
    #         for x in range(len(dataset) - 1):
    #             if random.random() < split:
    #                 self.__training_set.append(dataset[x])
    #             else:
    #                 self.__test_set.append(dataset[x])
    #         myfile.close()
    #
    # # Calculate the euclidean distance between every unigram feature vector in two datasets
    # def euclidean_distance(self, dataset1, dataset2, length):
    #     distance = 0
    #     for x in range(length):
    #         distance += pow((dataset1[x] - dataset2[x]), 2)
    #     return math.sqrt(distance)
    #
    # def get_neighbors(self, trainingSet, testInstance, k):
    #     distances = []
    #     length = len(testInstance) - 1
    #     for x in range(len(trainingSet)):
    #         dist = self.euclidean_distance(testInstance, trainingSet[x], length)
    #         distances.append((trainingSet[x], dist))
    #     distances.sort(key=operator.itemgetter(1))
    #     neighbors = []
    #     for x in range(k):
    #         neighbors.append(distances[x][0])
    #     return neighbors
    #
    # def get_response(self, neighbors):
    #     classVotes = {}
    #     for x in range(len(neighbors)):
    #         response = neighbors[x][1]
    #         if response in classVotes:
    #             classVotes[response] += 1
    #         else:
    #             classVotes[response] = 1
    #     sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    #     return sortedVotes[0][0]
    #
    # def getAccuracy(self, testSet, predictions):
    #     correct = 0
    #     for x in range(len(testSet)):
    #         if testSet[x][1] == predictions[x]:
    #             correct += 1
    #     return (correct / float(len(testSet))) * 100.0
    #
    # def k_nearest_neighbor(self):
    #     # generate predictions
    #     predictions = []
    #     k = 3
    #     for x in range(len(self.__test_set)):
    #         neighbors = self.get_neighbors(self.__training_set, self.__test_set[x], k)
    #         result = self.get_response(neighbors)
    #         predictions.append(result)
    #         print('> predicted=' + repr(result) + ', actual=' + repr(self.__test_set[x][1]))
    #     accuracy = self.getAccuracy(self.__test_set, predictions)
    #     print('Accuracy: ' + repr(accuracy) + '%')
