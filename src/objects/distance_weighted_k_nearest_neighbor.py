from .k_nearest_neighbor import KNearestNeighbor
import threading
import math
import operator

"""
 Subclass of KNearestNeighbor.

 Performs K-Nearest Neighbor on the dataset with
 weights based on distance during predictions. Classifier can be 1 or -1.
"""
class DistanceWeightedKNearestNeighbor(KNearestNeighbor):

    def __init__(self, machine_learner, k, is_global):
        super().__init__(machine_learner, k)
        self.__is_global = is_global

    @property
    def is_global(self):
        return self.__is_global

    def run(self):
        self.machine_learner.client.gui.machine_learner_window.display_message("\nDistance Weighted K-Nearest Neighbor initialized...")
        self.load_dataset()
        self.normalize_dataset()
        self.distance_weighted_k_nearest_neighbor(self.k)

    # Weight Formula for votes
    def get_weight(self, distance, weighting_exponent):
        if distance != 0:
            # 1/distance^weighting_exponent is the weight formula
            return 1/(pow(distance, weighting_exponent))
        # if distance is 0 then the weight would be one, maybe..
        else:
            return 1

    def get_distances(self, training_set, test_instance, k):
        distances = []
        length = len(test_instance) - 1
        # iterate through the training set
        for x in range(len(training_set)):
            # calculate the euclidean_distance of each instance to the test instance.
            dist = self.euclidean_distance(test_instance, training_set[x], length)
            distances.append((training_set[x], dist))
        # Sort the distances in ascending order
        distances.sort(key=operator.itemgetter(1))
        return distances;

    def get_prediction(self, distances):
        classVotes = {}
        # if you want to look at all training sets
        if self.is_global:
            for x in range(len(distances)):
                # get the distance of the training set to the test set
                distance = distances[x][1]
                # get the weight of the training set
                weight = self.get_weight(distance, 2)
                # get the training set's classification
                response = distances[x][0][1]

                if response in classVotes:
                    # add a weighted vote to the response thats already there
                    classVotes[response] += weight
                else:
                    # initialize the response in the class votes array to the weight
                    classVotes[response] = weight

            # Sort votes for class attribute in descending order
            sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

            # Return the prediction with the most votes
            return sortedVotes[0][0]
        else:
            neighbors = []
            neighborDistances = []
            for x in range(self.k):
                # iterate up to k and add each member of the training set that has the lowest distance to the test instance.
                neighbors.append(distances[x][0])
                neighborDistances.append(distances[x][1])

            # Iterate through all of the neighbors
            for x in range(len(neighbors)):
                weight = self.get_weight(neighborDistances[x], 2)
                # response is equal to each neighbor's class attribute
                response = neighbors[x][1]
                # if the response has already been added to class votes array
                if response in classVotes:
                    # add a vote to the response thats already there
                    classVotes[response] += weight
                else:
                    # initialize the response in the class votes array to 1
                    classVotes[response] = weight

            # Sort votes for class attribute in descending order
            sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

            # Return the prediction with the most votes
            return sortedVotes[0][0]

    # The actual distance weighted k-nearest neighbor algorithm.
    def distance_weighted_k_nearest_neighbor(self, k):
        """ Uses Leave One Out for Cross Validation """
        self.machine_learner.client.gui.machine_learner_window.display_message("\nStarting training...")
        predictions = []

        # Loop through dataset taking one set as test set and the rest as training sets
        for x in range(len(self.dataset)):
            test_set = self.dataset[x]
            # training set consists of every element in dataset except for the test set
            training_sets = [t for i,t in enumerate(self.dataset) if i!=x]

            distances = self.get_distances(training_sets, test_set, k)

            prediction = self.get_prediction(distances)

            # print('> predicted=' + repr(prediction) + ', actual=' + repr(test_set[1]))
            self.machine_learner.client.gui.machine_learner_window.display_message('\npredicted=' + repr(prediction) + ', actual=' + repr(test_set[1]))

            # add the neighbors' prediction to the predictions array
            predictions.append(prediction)

        # get the classification accuracy for all of the predictions
        accuracy = self.getAccuracy(self.dataset, predictions)

        self.machine_learner.client.gui.machine_learner_window.display_message('\nAccuracy: ' + repr(accuracy) + '%')
        print('Accuracy: ' + repr(accuracy) + '%')
