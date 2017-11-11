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

    def run(self):
        self.grnn = GeneralRegressionNeuralNetwork(self.client)
        self.grnn.start()
        self.grnn.join()

    def steady_state_genetic_algorithm(self):
        print("Running SSGA..")
        random.seed()
        self.create_population(20)
        generation = 0
        while generation != 180:

            mother, mother_index, mother_fitness = self.tournament_selection(self.population, 3)
            father, father_index, father_fitness = self.tournament_selection(self.population, 3)

            first_child, second_child = self.cross_blx(0.0, mother, father)

            first_child_fitness = self.get_fitness(first_child)
            second_child_fitness = self.get_fitness(second_child)

            fitnesses = [[mother, mother_fitness], [father, father_fitness], [first_child, first_child_fitness], [second_child, second_child_fitness]]

            fitnesses.sort(key=operator.itemgetter(1), reverse=False)

            self.population[mother_index] = fitnesses[0][0]
            self.population[father_index] = fitnesses[1][0]

            generation = generation + 1
            print("Generation #" + repr(generation) + " Best Fitness: " + repr(fitnesses[0][1]))
            # print(fitnesses[0][1])
            # if fitnesses[0][1] == 1:
            #     print("Found a fitness that is 1." + repr(self.population))
            #     break
        print('Prediction Population[0]: ' + repr(self.grnn.single(self.population[0])))
        print('Prediction Population[1]: ' + repr(self.grnn.single(self.population[1])))
        print('Prediction Population[2]: ' + repr(self.grnn.single(self.population[2])))

        return self.population

            # print("Child #1 Selected: " + repr(self.population[mother_index]))
            # print("Child #2 Selected: " + repr(self.population[father_index]))

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
        best_fitness = 0
        parent_index = 0
        for x in range(k):
            index = random.randrange(0, len(population))
            fitness = self.get_fitness(population[index])

            if fitness > best_fitness:
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
