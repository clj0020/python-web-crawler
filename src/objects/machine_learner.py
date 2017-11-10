import threading
import json
import operator
import xlsxwriter
from .k_nearest_neighbor import KNearestNeighbor
from .distance_weighted_k_nearest_neighbor import DistanceWeightedKNearestNeighbor
from .general_regression_neural_network import GeneralRegressionNeuralNetwork


ENCODING = 'utf-8'

class MachineLearner(threading.Thread):

    def __init__(self, client):
        super().__init__(daemon=True, target=self.run)
        self.__client = client
        self.__k_nearest_neighbor = None
        self.__distance_weighted_k_nearest_neighbor = None
        self.__grnn = None


    def run(self):
        print("Machine Learner initialized...")

    @property
    def k_nearest_neighbor(self):
        return self.__k_nearest_neighbor

    @property
    def distance_weighted_k_nearest_neighbor(self):
        return self.__distance_weighted_k_nearest_neighbor

    @property
    def grnn(self):
        return self.__grnn

    @property
    def client(self):
        return self.__client

    def initialize_k_nearest_neighbor(self, k):
        self.__k_nearest_neighbor = KNearestNeighbor(self.client, k)
        self.__k_nearest_neighbor.start()
        self.__k_nearest_neighbor.join()
        self.__k_nearest_neighbor.k_nearest_neighbor(k)

    def evaluate_k_nearest(self):
        self.__k_nearest_neighbor = KNearestNeighbor(self.client, 0)
        self.__k_nearest_neighbor.start()
        self.__k_nearest_neighbor.join()
        accuracies = self.__k_nearest_neighbor.evaluate_k()
        accuracies_string = json.dumps(accuracies)

        # message = 'machine_learner;' + 'show_accuracies_chart;' + accuracies_string
        # self.client.queue.put(message.encode(ENCODING))

        self.write_to_excel(accuracies, 'k-nearest-manhattan-accuracies.xlsx')

        accuracies.sort(key=operator.itemgetter(1), reverse=True)

        print("Best K Value: {}".format(accuracies[0]))
        print("Second Best K Value: {}".format(accuracies[1]))
        print("Third Best K Value: {}".format(accuracies[2]))
        print("Fourth Best K Value: {}".format(accuracies[3]))

    def initialize_distance_weighted_k_nearest_neighbor(self, k, is_global):
        self.__distance_weighted_k_nearest_neighbor = DistanceWeightedKNearestNeighbor(self.client, k, is_global)
        self.__distance_weighted_k_nearest_neighbor.start()
        self.__distance_weighted_k_nearest_neighbor.join()
        self.__distance_weighted_k_nearest_neighbor.distance_weighted_k_nearest_neighbor(k)


    def evaluate_distance_weighted_k_nearest(self, is_global):
        self.__distance_weighted_k_nearest_neighbor = DistanceWeightedKNearestNeighbor(self.client, 0, is_global)
        self.__distance_weighted_k_nearest_neighbor.start()
        self.__distance_weighted_k_nearest_neighbor.join()
        accuracies = self.__distance_weighted_k_nearest_neighbor.evaluate_k()

        accuracies_string = json.dumps(accuracies)

        # message = 'machine_learner;' + 'show_distance_accuracies_chart;' + accuracies_string
        # self.client.queue.put(message.encode(ENCODING))

        self.write_to_excel(accuracies, 'local-distance-weighted-k-nearest-manhattan-accuracies.xlsx')

        accuracies.sort(key=operator.itemgetter(1), reverse=True)

        print("Best K Value (distance weighted): {}".format(accuracies[0]))
        print("Second Best K Value (distance weighted): {}".format(accuracies[1]))
        print("Third Best K Value (distance weighted): {}".format(accuracies[2]))
        print("Fourth Best K Value (distance weighted): {}".format(accuracies[3]))

    def initialize_grnn(self):
        self.__grnn = GeneralRegressionNeuralNetwork(self.client)
        self.__grnn.start()
        self.__grnn.join()
        self.__grnn.train()

    def write_to_excel(self, accuracies, filename):
        workbook = xlsxwriter.Workbook(filename)
        worksheet = workbook.add_worksheet()

        row = 0

        for col, data in enumerate(accuracies):
            worksheet.write_column(row, col, data)

        workbook.close()
