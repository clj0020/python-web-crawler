import threading
import requests
from bs4 import BeautifulSoup
import uuid, base64
import io
from .tree import Tree
from .webpage_classifier import WebpageClassifier
from .steady_state_genetic import SteadyStateGenetic
from .general_regression_neural_network import GeneralRegressionNeuralNetwork
import numpy as np

class WebScraper(threading.Thread):

    def __init__(self, client, url, depth):
        super().__init__(daemon=True, target=self.run)
        self.client = client
        self.url = url
        self.depth = depth
        self.tree = Tree(self)
        self.webpage_classifier = WebpageClassifier(self.client)
        self.num_websites_scraped = 0
        self.num_sites_in_interval = 0
        self.predictions = []
        self.population = []

    def run(self):
        print("Web Scraper initialized.")
        self.tree.add_node(self.url)
        self.iterative_deepening_search(self.url, self.depth)

    def depth_limited_search(self, node, depth):

        def recursive_depth_limited_search(node, depth):
            # experiment_one = threading.Thread(target=self.experiment_one, args=[self.tree[node]], daemon=True)
            # experiment_one.start()
            # experiment_one.join()
            #
            # unigram_vector = self.experiment_one(self.tree[node])
            #
            # if unigram_vector is None:
            #     return 'cutoff'
            #

            experiment_two = threading.Thread(target=self.experiment_two, args=[self.tree[node]], daemon=True)
            experiment_two.start()
            experiment_two.join()


            children = self.tree[node].find_children()

            if depth == 0:
                return 'cutoff'
            else:
                cutoff_occurred = False
                for child in children:
                    result = recursive_depth_limited_search(child, depth - 1)
                    if result == 'cutoff':
                        cutoff_occurred = True
                    elif result is not None:
                        return result
                return 'cutoff' if cutoff_occurred else None

        return recursive_depth_limited_search(node, depth)

    def iterative_deepening_search(self, rootNode, maxDepth):
        for depth in range(maxDepth):
            result = self.depth_limited_search(rootNode, depth)
            if result != 'cutoff':
                self.client.gui.display_message('\nWebsites Added: ' + repr(self.websites_added))
                print('\nWebsites Added: ' + repr(self.websites_added))
                return result

    def experiment_one(self, site):
        file_save = threading.Thread(target=site.save_file, daemon=True)
        file_save.start()
        file_save.join()


        unigram_vector = self.webpage_classifier.scrape_site(site)

        if unigram_vector is None:
            return None

        self.num_websites_scraped = self.num_websites_scraped + 1
        self.predictions.append(unigram_vector[1])

        self.client.gui.display_message(repr(self.num_websites_scraped) + " sites scraped.")

        if unigram_vector[1] >= -0.015 and unigram_vector[1] <= 0.015:
            self.num_sites_in_interval = self.num_sites_in_interval + 1
            print(site.identifier)

        if self.num_websites_scraped == 100:
            print("100 websites scraped!")

            standard_deviation_first = np.std(self.predictions)
            mean_first = np.mean(self.predictions)

            self.client.gui.display_message("Standard Deviation of First 100 sites scraped: " + repr(standard_deviation_first))
            self.client.gui.display_message("Mean of First 100 sites scraped: " + repr(mean_first))
            print("Standard Deviation of First 100 sites scraped: " + repr(standard_deviation_first))
            print("Mean of First 100 sites scraped: " + repr(mean_first))

        if self.num_websites_scraped == 200:
            print("200 sites scraped!")

            standard_deviation_last = np.std(self.predictions[99:200])
            mean_last = np.mean(self.predictions[99:200])

            self.client.gui.display_message("Standard Deviation of Last 100 sites scraped: " + repr(standard_deviation_last))
            self.client.gui.display_message("Mean of Last 100 sites scraped: " + repr(mean_last))
            print("Standard Deviation of Last 100 sites scraped: " + repr(standard_deviation_last))
            print("Mean of Last 100 sites scraped: " + repr(mean_last))

            standard_deviation_overall = np.std(self.predictions)
            mean_overall = np.mean(self.predictions)

            self.client.gui.display_message("Standard Deviation of 200 sites scraped: " + repr(standard_deviation_overall))
            self.client.gui.display_message("Mean of 200 sites scraped: " + repr(mean_overall))
            print("Standard Deviation of 200 sites scraped: " + repr(standard_deviation_overall))
            print("Mean of 200 sites scraped: " + repr(mean_overall))

            self.client.gui.display_message("Number of Sites in the interval [-0.015, 0.015]: " + repr(self.num_sites_in_interval))
            print("Number of Sites in the interval [-0.015, 0.015]: " + repr(self.num_sites_in_interval))

        return unigram_vector

    def experiment_two(self, site):
        file_save = threading.Thread(target=site.save_file, daemon=True)
        file_save.start()
        file_save.join()

        frequency = site.frequency

        if frequency is None:
            return None

        # extract the chars and their unigram_vector into two lists
        chars, unigram_vector = map(list,zip(*frequency))

        sum = 0
        for x in range(len(unigram_vector)):
            sum = unigram_vector[x] + sum

        for x in range(len(unigram_vector)):
            unigram_vector[x] = unigram_vector[x] / sum

        unigram_vector = self.webpage_classifier.normalize_dataset(unigram_vector)

        classification = 0 # temp

        unigram_vector.insert(0, classification)
        unigram_vector.insert(0, 0)

        self.num_websites_scraped = self.num_websites_scraped + 1

        self.population.append(unigram_vector)

        # if len(self.population) == 20:
        #     # Run the steady state genetic algorithm on the population
        #     ssga = SteadyStateGenetic(self.client, self.population)
        #     ssga.start()
        #     ssga.join()
        #
        #     self.population = ssga.steady_state_genetic_algorithm()
        #
        #     grnn = GeneralRegressionNeuralNetwork(self.client)
        #     grnn.start()
        #     grnn.join()
        #
        #     prediction = grnn.single(self.population[])
        #
        #     for x in range(len(self.population)):
        #         prediction = grnn.single(self.population[x])
        #         print(prediction)
        #
        #     # Reset list
        #     self.population = []
