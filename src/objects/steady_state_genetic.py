import threading
import random
import datetime
import operator
import decimal
import math
import numpy as np
from .general_regression_neural_network import GeneralRegressionNeuralNetwork

class SteadyStateGenetic(threading.Thread):

    def __init__(self, client):
        super().__init__(daemon=True, target=self.run)
        self.client = client
        self.population = []
        self.overall_population = []

    def run(self):
        self.grnn = GeneralRegressionNeuralNetwork(self.client)
        self.grnn.start()
        self.grnn.join()

    def steady_state_genetic_algorithm(self):
        print("Running SSGA..")
        random.seed()
        self.create_population(20)
        generation = 0
        generations = []
        while generation != 200:

            mother, mother_index, mother_fitness = self.tournament_selection(self.population, 2)
            father, father_index, father_fitness = self.tournament_selection(self.population, 2)

            first_child, second_child = self.cross_blx(0.0, mother, father)

            first_child_fitness = self.get_fitness(first_child)
            second_child_fitness = self.get_fitness(second_child)

            fitnesses = [[first_child, first_child_fitness], [second_child, second_child_fitness]]

            fitnesses.sort(key=operator.itemgetter(1), reverse=False)

            worst_fitness = 0
            best_fitness = 1
            for x in range(len(self.population)):
                fitness = self.get_fitness(self.population[x])
                if fitness > worst_fitness:
                    least_fit = self.population[x]
                    least_fit_index = x
                    worst_fitness = fitness
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_fitness_item = self.population[x]

            self.population[least_fit_index] = fitnesses[0][0]

            generation = generation + 1
            print("Generation #" + repr(generation) + " Best Fitness: " + repr(best_fitness))
            generations.append([generation, best_fitness])

            self.overall_population.append(best_fitness_item)

        self.population = []
        return generations

    def cross_blx(self, alpha, mother, father):
        first_child = []
        second_child = []

        for i in range(len(mother)):
            distance = abs(mother[i] - father[i])
            minimum = min(mother[i], father[i]) - alpha * distance
            maximum = max(mother[i], father[i]) + alpha * distance

            first_child.append(random.uniform(minimum, maximum))
            second_child.append(random.uniform(minimum, maximum))

        return first_child, second_child

    def tournament_selection(self, population, k):
        """ Pick K amount of chromosomes out of the parent and then return the best one. """
        best_fitness = 1
        parent_index = 0
        for x in range(k):
            index = random.randrange(0, len(population))
            fitness = self.get_fitness(population[index])

            if fitness < best_fitness:
                best_fitness = fitness
                parent = population[index]
                parent_index = index
        return parent, parent_index, best_fitness

    def get_fitness(self, chromosome):
        prediction = self.grnn.single(chromosome)
        return abs(prediction)

    def create_population(self, population_size):
        for x in range(population_size):
            unigram_vector = []
            for y in range(95):
                # feature = float(decimal.Decimal(random.randrange(0, 1))/100)
                feature = float(round(random.uniform(0.0, 1.0), 5))
                unigram_vector.append(feature)
            unigram_vector.insert(0, 1.0)
            unigram_vector.insert(0, 1.0)
            self.population.append(unigram_vector)

        self.normalize_population()
        # self.overall_population.append(self.population)

    def normalize_population(self):
        for x in range(len(self.population)):
            magnitude = self.get_magnitude(self.population[x])
            for y in range(2, 97):
                if magnitude != 0:
                    self.population[x][y] = self.population[x][y] / magnitude

    def get_magnitude(self, vector):
        magnitude = 0
        for x in range(2, 97):
            magnitude += pow(vector[x], 2)
        return math.sqrt(magnitude)

    def evaluate_overall_population(self):
        print("# of Vectors in Overall Population: " + repr(len(self.overall_population)))

        output_array = []
        num_sites_in_interval = 0
        predictions = []
        for x in range(len(self.overall_population)):
            prediction = self.grnn.single(self.overall_population[x])
            predictions.append(prediction)

            if prediction >= -0.015 and prediction <= 0.015:
                num_sites_in_interval = num_sites_in_interval + 1

        standard_deviation_first = np.std(predictions[0:99])
        mean_first = np.mean(predictions[0:99])

        output_array.append(['standard_deviation_first', standard_deviation_first])
        output_array.append(['mean_first', mean_first])

        print("Standard Deviation of First 100 sites scraped: " + repr(standard_deviation_first))
        print("Mean of First 100 sites scraped: " + repr(mean_first))

        standard_deviation_last = np.std(predictions[100:199])
        mean_last = np.mean(predictions[100:199])

        output_array.append(['standard_deviation_last', standard_deviation_last])
        output_array.append(['mean_last', mean_last])

        print("Standard Deviation of Last 100 sites scraped: " + repr(standard_deviation_last))
        print("Mean of Last 100 sites scraped: " + repr(mean_last))


        standard_deviation_overall = np.std(predictions)
        mean_overall = np.mean(predictions)

        output_array.append(['standard_deviation_overall', standard_deviation_overall])
        output_array.append(['mean_overall', mean_overall])

        print("Standard Deviation of 200 sites scraped: " + repr(standard_deviation_overall))
        print("Mean of 200 sites scraped: " + repr(mean_overall))

        print("Number of Sites in the interval [-0.015, 0.015]: " + repr(num_sites_in_interval))

        output_array.append(['num_sites_in_interval', num_sites_in_interval])

        self.overall_population = []

        return output_array
