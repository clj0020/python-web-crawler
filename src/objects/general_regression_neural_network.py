import threading
import math
import operator
import numpy as np

class GeneralRegressionNeuralNetwork(threading.Thread):

    def __init__(self, machine_learner):
        super().__init__(daemon=False, target=self.run)
        self.__dataset = []
        self.__machine_learner = machine_learner

    @property
    def dataset(self):
        return self.__dataset

    @property
    def machine_learner(self):
        return self.__machine_learner

    def run(self):
        self.load_dataset()
        self.normalize_dataset()
        # sigma = self.d_max()
        training_sets, classification_sets = self.split_array()

        data = training_sets[:]
        self.machine_learner.client.gui.machine_learner_window.display_message("\nRunning GRNN...")
        predictions = []
        for x in data:
            # result = self.grnn(x, training_sets, classification_sets, sigma)[0]
            result = self.grnn(x, training_sets, classification_sets, 0.008574707368937534)[0]

            if result < 0:
                predictions.append(-1)
            else:
                predictions.append(1)

        correct = 0
        truePositive = 0
        falsePositive = 0
        falseNegative = 0
        trueNegative = 0
        for x in range(len(self.dataset)):
            actualValue = self.dataset[x][1]
            prediction = predictions[x]

            if prediction == 1 and actualValue == 1:
                correct += 1
                truePositive += 1
            elif prediction == 1 and actualValue == -1:
                falsePositive += 1
            elif prediction == -1 and actualValue == -1:
                correct += 1
                trueNegative += 1
            elif prediction == -1 and actualValue == 1:
                falseNegative += 1

            # if actualValue == prediction:
            #     correct += 1

        accuracy = (correct / float(len(self.dataset))) * 100.0

        print(accuracy)
        print("Correct: " + repr(correct))
        self.machine_learner.client.gui.machine_learner_window.display_message("\nAccuracy: " + repr(accuracy) + "%")
        self.machine_learner.client.gui.machine_learner_window.display_message("\nTrue Positives: " + repr(truePositive))
        self.machine_learner.client.gui.machine_learner_window.display_message("\nFalse Positives: " + repr(falsePositive))
        self.machine_learner.client.gui.machine_learner_window.display_message("\nTrue Negatives: " + repr(trueNegative))
        self.machine_learner.client.gui.machine_learner_window.display_message("\nFalse Negatives: " + repr(falseNegative))

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

    def split_array(self):
        training_sets = []
        classification_sets = []
        for x in range(len(self.dataset)):
            training_sets.append(self.dataset[x][2:97])
            classification_sets.append([self.dataset[x][1]])
        return training_sets, classification_sets

    def get_magnitude(self, vector):
        magnitude = 0
        for x in range(2, 97):
            magnitude += pow(vector[x], 2)
        return math.sqrt(magnitude)

    # Normalize the unigram vectors from the dataset
    def normalize_dataset(self):
        self.machine_learner.client.gui.machine_learner_window.display_message("\nNormalizing dataset...")
        for x in range(len(self.dataset)):
            magnitude = self.get_magnitude(self.dataset[x])
            # set each unigram value as
            for y in range(2, 97):
                if magnitude != 0:
                    self.__dataset[x][y] = self.dataset[x][y] / magnitude

    def manhattan_distance(self, dataset1, dataset2):
        distance = 0
        # for all unigram values in the unigram vectors
        for x in range(2, 97):
            # add the absolute value of the two values subtracted
            distance += abs(dataset1[x] - dataset2[x])
        return distance

    # Calculate the euclidean distance between every unigram feature vector in two datasets
    def euclidean_distance(self, dataset1, dataset2):
        distance = 0
        # for all unigram values in the unigram vectors
        for x in range(2, 97):
            distance += pow((dataset1[x] - dataset2[x]), 2)
        return math.sqrt(distance)

    def d_max(self):
        self.machine_learner.client.gui.machine_learner_window.display_message("\nFinding sigma...")
        distances = []
        for x in range(len(self.dataset) - 1):
            test_set = self.dataset[x]

            # training set consists of every element in dataset except for the test set
            training_sets = [t for i,t in enumerate(self.dataset) if i!=x]

            for y in range(len(training_sets) - 1):
                dist = self.euclidean_distance(test_set, training_sets[y])
                distances.append((training_sets[y], dist))

        # Sort the distances in ascending order
        distances.sort(key=operator.itemgetter(1), reverse=True)

        self.machine_learner.client.gui.machine_learner_window.display_message("\nSigma=" + repr(distances[0][1]))
        return distances[0][1];

    def activator(self, data, train_x, sigma):
        distance = 0
        for i in range(len(data)):

            distance += math.pow((data[i] - train_x[i]), 2)
        # distance = (math.sqrt(distance))
        # return math.exp(- math.pow(distance, 2) / (2* math.pow(sigma, 2)))
        return math.exp(- (math.pow(distance, 2) / (2 * (math.pow(sigma, 2)))))

    def grnn(self, data, training_set, classification_array, sigma):
        result = []
        out_dim = len(classification_array[1])
        for dim in range(out_dim):
            factor, divide = 0, 0
            for i in range(len(training_set)):
                cache = self.activator(data, training_set[i], sigma)
                factor += classification_array[i][dim] * cache
                divide += cache
            result.append(factor/divide)
        return result
