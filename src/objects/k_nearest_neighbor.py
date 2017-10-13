import threading
import math
import operator
import random

class KNearestNeighbor(threading.Thread):

    def __init__(self, machine_learner, k, split):
        super().__init__(daemon=False, target=self.run)
        self.__machine_learner = machine_learner
        self.__training_set = []
        self.__test_set = []
        self.__split = split
        self.__k = k


    def run(self):
        self.machine_learner.client.gui.machine_learner_window.display_message("\nKNearestNeighbor initialized...")
        self.load_dataset(self.split)
        self.k_nearest_neighbor(self.k)

    @property
    def machine_learner(self):
        return self.__machine_learner

    @property
    def training_set(self):
        return self.__training_set

    @property
    def test_set(self):
        return self.__test_set

    @property
    def k(self):
        return self.__k

    @property
    def split(self):
        return self.__split

    # Load the data into the training and test sets from the dataset file
    def load_dataset(self, split):
        self.machine_learner.client.gui.machine_learner_window.display_message("\nLoading dataset...")
        with open('datasets/our_dataset.txt') as myfile:
            lines = myfile.readlines()
            dataset = []

            # Create a 2d array with the numbers in the dataset file
            for line in lines:
                array = []
                for number in line.split():
                    array.append(float(number))
                dataset.append(array)

            # Randomly add datasets to either test set or training set based on split
            for x in range(len(dataset) - 1):
                if random.random() < split:
                    self.__training_set.append(dataset[x])
                else:
                    self.__test_set.append(dataset[x])
            myfile.close()

    # Calculate the euclidean distance between every unigram feature vector in two datasets
    def euclidean_distance(self, dataset1, dataset2, length):
        distance = 0
        for x in range(length):
            distance += pow((dataset1[x] - dataset2[x]), 2)
        return math.sqrt(distance)

    # Get the k amount of neighbors of a training set and a test instance
    def get_neighbors(self, trainingSet, testInstance, k):
        distances = []
        length = len(testInstance) - 1
        # iterate through the training set
        for x in range(len(trainingSet)):
            # calculate the euclidean_distance of each instance to the test instance.
            dist = self.euclidean_distance(testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))
        # Sort the distances in ascending order
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            # iterate up to k and add each member of the training set that has the lowest distance to the test instance.
            neighbors.append(distances[x][0])
        return neighbors

    # Get the prediction based on the neighbors
    # Get each neighbor to vote for what they think their class attribute (malicious: -1 or safe: 1) is and then take the majority vote as the prediction
    def get_response(self, neighbors):
        classVotes = {}
        # Iterate through all of the neighbors
        for x in range(len(neighbors)):
            # response is equal to each neighbor's class attribute
            response = neighbors[x][1]
            # if the response has already been added to class votes array
            if response in classVotes:
                # add a vote to the response thats already there
                classVotes[response] += 1
            else:
                # initialize the response in the class votes array to 1
                classVotes[response] = 1

        # Sort votes for class attribute in descending order
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

        # Return the prediction with the most votes
        return sortedVotes[0][0]

    # Get the accuracy of the predictions (classification accuracy) by calculating the
    # ratio of total correct predictions out of all predictions made.
    def getAccuracy(self, testSet, predictions):
        correct = 0
        # iterate through the test set
        for x in range(len(testSet)):
            # if the classification of the test set matches the prediction
            if testSet[x][1] == predictions[x]:
                correct += 1 # add 1 to correct
        # return the result of dividing the number of correct predictions by the total amount of predictions and multiplying by 100
        return (correct / float(len(testSet))) * 100.0

    # The actual k-nearest neighbor algorithm
    def k_nearest_neighbor(self, k):
        self.machine_learner.client.gui.machine_learner_window.display_message("\nStarting training...")

        predictions = []
        # iterate through the test set
        for x in range(len(self.__test_set)):
            # find the k amount of neighbors in the training set for each instance of the test set
            neighbors = self.get_neighbors(self.__training_set, self.__test_set[x], k)

            # get the neighbors' prediction
            result = self.get_response(neighbors)

            # add the neighbors' prediction to the predictions array
            predictions.append(result)
            # self.machine_learner.client.gui.machine_learner_window.display_message('\n> predicted=' + repr(result) + ', actual=' + repr(self.__test_set[x][1]))
            print('> predicted=' + repr(result) + ', actual=' + repr(self.__test_set[x][1]))

        # get the classification accuracy for all of the predictions
        accuracy = self.getAccuracy(self.__test_set, predictions)
        self.machine_learner.client.gui.machine_learner_window.display_message('\nAccuracy: ' + repr(accuracy) + '%')
        print('Accuracy: ' + repr(accuracy) + '%')
