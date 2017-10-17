import threading
import math
import operator
import random
from random import sample

class KNearestNeighbor(threading.Thread):

    def __init__(self, machine_learner, k):
        super().__init__(daemon=False, target=self.run)
        self.__machine_learner = machine_learner
        self.__dataset = []
        self.__k = k


    def run(self):
        self.machine_learner.client.gui.machine_learner_window.display_message("\nK Nearest Neighbor initialized...")
        self.load_dataset()
        self.k_nearest_neighbor(self.k)

    @property
    def machine_learner(self):
        return self.__machine_learner

    @property
    def dataset(self):
        return self.__dataset

    @property
    def k(self):
        return self.__k

    # Load the data into the training and test sets from the dataset file
    def load_dataset(self):
        self.machine_learner.client.gui.machine_learner_window.display_message("\nLoading dataset...")
        with open('datasets/our_dataset.txt') as myfile:
            lines = myfile.readlines()

            # Create a 2d array with the numbers in the dataset file
            for line in lines:
                array = []
                for number in line.split():
                    array.append(float(number))
                self.__dataset.append(array)

            myfile.close()

    # Calculate the euclidean distance between every unigram feature vector in two datasets
    def euclidean_distance(self, dataset1, dataset2, length):
        distance = 0
        # for all unigram values in the unigram vectors
        for x in range(2, 97):
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
    def getAccuracy(self, dataset, predictions):
        correct = 0
        # iterate through the test set
        for x in range(len(dataset)):
            # if the classification of the test set matches the prediction
            if dataset[x][1] == predictions[x]:
                correct += 1 # add 1 to correct

        # return the result of dividing the number of correct predictions by the total amount of predictions and multiplying by 100
        return (correct / float(len(dataset))) * 100.0

    # The actual k-nearest neighbor algorithm.
    def k_nearest_neighbor(self, k):
        """ Uses Leave One Out for Cross Validation """
        self.machine_learner.client.gui.machine_learner_window.display_message("\nStarting training...")
        predictions = []

        # Loop through dataset taking one set as test set and the rest as training sets
        for x in range(len(self.dataset)):
            test_set = self.dataset[x]
            # training set consists of every element in dataset except for the test set
            training_sets = [t for i,t in enumerate(self.dataset) if i!=x]

            # Get the k nearest neighbors of the test_set in the training sets
            neighbors = self.get_neighbors(training_sets, test_set, k)

            # get the neighbors' majority vote for a prediction
            prediction = self.get_response(neighbors)
            print('> predicted=' + repr(prediction) + ', actual=' + repr(test_set[1]))
            self.machine_learner.client.gui.machine_learner_window.display_message('\npredicted=' + repr(prediction) + ', actual=' + repr(test_set[1]))

            # add the neighbors' prediction to the predictions array
            predictions.append(prediction)

        # get the classification accuracy for all of the predictions
        accuracy = self.getAccuracy(self.dataset, predictions)

        self.machine_learner.client.gui.machine_learner_window.display_message('\nAccuracy: ' + repr(accuracy) + '%')
        print('Accuracy: ' + repr(accuracy) + '%')
