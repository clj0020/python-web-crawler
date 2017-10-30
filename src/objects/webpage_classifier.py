import threading
import math
from .web_page import WebPage
from .distance_weighted_k_nearest_neighbor import DistanceWeightedKNearestNeighbor

class WebpageClassifier(threading.Thread):

    def __init__(self, client):
        super().__init__(daemon=True, target=self.run)
        self.__client = client

    def run(self):
        print("Webpage Classifier initialized...")

    @property
    def client(self):
        return self.__client

    def scrape_site(self, url):
        print("Scraping site: {}".format(url))
        webpage = WebPage(self.client, url)
        frequency = webpage.frequency

        print(len(frequency))

        # extract the chars and their counts into two lists
        chars, counts = map(list,zip(*frequency))

        self.normalize_dataset(counts)

        # add two numbers in front of array to symbolize dataset number and classification
        datasetNumber = 599
        classification = 0 # temp

        counts.insert(0, classification)
        counts.insert(0, datasetNumber)

        distance_weighted_k_nearest_neighbor = DistanceWeightedKNearestNeighbor(self.client, 6, False)
        distance_weighted_k_nearest_neighbor.start()
        distance_weighted_k_nearest_neighbor.join()
        distance_weighted_k_nearest_neighbor.distance_weighted_k_nearest_neighbor_single(6, counts)

    def get_magnitude(self, vector):
        magnitude = 0
        for x in range(len(vector)):
            magnitude += pow(vector[x], 2)
        return math.sqrt(magnitude)

    # Normalize the unigram vectors from the dataset
    def normalize_dataset(self, dataset):
        magnitude = self.get_magnitude(dataset)
        for x in range(len(dataset)):
            if magnitude != 0:
                dataset[x] = dataset[x] / magnitude
