import threading
import math
import operator
import numpy as np
from decimal import Decimal

class KNearestNeighbor(threading.Thread):

    def __init__(self, client, k):
        super().__init__(daemon=True, target=self.run)
        self.__client = client
        self.__dataset = []
        self.__k = k



    def run(self):
        self.client.gui.display_message("\nK Nearest Neighbor initialized...")
        self.load_dataset()
        self.normalize_dataset()
        # self.k_nearest_neighbor(self.k)
        # self.evaluate_k()

    @property
    def client(self):
        return self.__client

    @property
    def dataset(self):
        return self.__dataset

    @property
    def k(self):
        return self.__k

    # Load the data into the training and test sets from the dataset file
    def load_dataset(self):
        if self.client is not None:
            self.client.gui.display_message("\nLoading dataset...")
        with open('datasets/our_dataset.txt') as myfile:
            lines = myfile.readlines()

            # Create a 2d array with the numbers in the dataset file
            for line in lines:
                array = []
                for number in line.split():
                    array.append(float(number))
                self.__dataset.append(array)

            myfile.close()

    def get_magnitude(self, vector):
        magnitude = 0
        for x in range(2, 97):
            magnitude += pow(vector[x], 2)
        return math.sqrt(magnitude)

    # Normalize the unigram vectors from the dataset
    def normalize_dataset(self):
        if self.client is not None:
            self.client.gui.display_message("\nNormalizing Data...")
        for x in range(len(self.dataset)):
            magnitude = self.get_magnitude(self.dataset[x])
            # set each unigram value as
            for y in range(2, 97):
                if magnitude != 0:
                    self.__dataset[x][y] = self.dataset[x][y] / magnitude

    # Calculate the euclidean distance between every unigram feature vector in two datasets
    def euclidean_distance(self, dataset1, dataset2, length):
        distance = 0
        # for all unigram values in the unigram vectors
        for x in range(2, 97):
            distance += pow((dataset1[x] - dataset2[x]), 2)
        return math.sqrt(distance)

    def manhattan_distance(self, dataset1, dataset2):
        distance = 0
        # for all unigram values in the unigram vectors
        for x in range(2, 97):
            # add the absolute value of the two values subtracted
            distance += abs(dataset1[x] - dataset2[x])
        return distance

    def square_rooted(self, x):
        return round(math.sqrt(sum([a*a for a in x])),3)

    def cosine_similarity(self, dataset1, dataset2):
        numerator = sum(a*b for a,b in zip(dataset1, dataset2))
        denominator = self.square_rooted(dataset1)*self.square_rooted(dataset2)
        return round(numerator/float(denominator),3)

    def nth_root(self, value, n_root):
        root_value = 1/float(n_root)
        return round (Decimal(value) ** Decimal(root_value),3)

    def minkowski_distance(self, dataset1, dataset2, p_value):
        return self.nth_root(sum(pow(abs(a-b), p_value) for a, b in zip(dataset1, dataset2)), p_value)

    # Get the k amount of neighbors of a training set and a test instance
    def get_neighbors(self, training_set, test_instance, k):
        distances = []
        length = len(test_instance) - 1
        # iterate through the training set
        for x in range(len(training_set)):
            # calculate the euclidean_distance of each instance to the test instance.
            dist = self.euclidean_distance(test_instance, training_set[x], length)

            # calculate the manhattan_distance of each instance to the test instance.
            # dist = self.manhattan_distance(test_instance, training_set[x])

            # calculate the cosine_similarity of each instance to the test instance.
            # dist = self.cosine_similarity(test_instance, training_set[x])

            # # calculate the minkowski_distance of each instance to the test instance.
            # dist = self.minkowski_distance(test_instance, training_set[x], 3)

            distances.append((training_set[x], dist))
        # Sort the distances in ascending order
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            # iterate up to k and add each member of the training set that has the lowest distance to the test instance.
            neighbors.append(distances[x][0])

        return neighbors

    # Get the prediction based on the neighbors
    # Get each neighbor to vote for what they think their class attribute (malicious: -1 or safe: 1) is and then take the majority vote as the prediction
    def get_prediction(self, neighbors):
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
        self.client.gui.display_message("\nStarting training...")
        predictions = []

        # Loop through dataset taking one set as test set and the rest as training sets
        for x in range(len(self.dataset)):
            test_set = self.dataset[x]
            # training set consists of every element in dataset except for the test set
            training_sets = [t for i,t in enumerate(self.dataset) if i!=x]

            # Get the k nearest neighbors of the test_set in the training sets
            neighbors = self.get_neighbors(training_sets, test_set, k)

            # get the neighbors' majority vote for a prediction
            prediction = self.get_prediction(neighbors)
            # print('> predicted=' + repr(prediction) + ', actual=' + repr(test_set[1]))
            self.client.gui.display_message('\npredicted=' + repr(prediction) + ', actual=' + repr(test_set[1]))

            # add the neighbors' prediction to the predictions array
            predictions.append(prediction)

        # get the classification accuracy for all of the predictions
        accuracy = self.getAccuracy(self.dataset, predictions)

        self.client.gui.display_message('\nAccuracy: ' + repr(accuracy) + '%' + ' for k=' + repr(k))
        print('Accuracy: ' + repr(accuracy) + '%' + ' KNN')
        return accuracy

    # Find the most efficient K value.
    def evaluate_k(self):
        self.client.gui.display_message("\nEvaluating the best K value...\nThis might take a while...")
        accuracies = []
        for x in range(1, 101):
            accuracy = self.k_nearest_neighbor(x)
            accuracies.append((x, accuracy))

        return accuracies
