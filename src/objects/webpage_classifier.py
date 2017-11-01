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

        sum = 0
        for x in range(len(counts)):
            sum = counts[x] + sum

        for x in range(len(counts)):
            counts[x] = counts[x] / sum

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

    # For an array of website urls and their classifications, scrape each site and add it's classification to the datasets file
    def add_webpages_to_dataset(self, websites):
        # number of datasets
        count = 634

        # extract the urls and their classifications into two lists
        urls, classifications = map(list,zip(*websites))

        for x in range(len(urls)):
            webpage = WebPage(self.client, urls[x])
            self.add_webpage_to_dataset(webpage, classifications[x], count)
            count = count + 1


    # Add a webpages unigram vector and its classification to the end of the dataset
    # Params: Webpage, Classification, Count (The number of datasets in the file)
    def add_webpage_to_dataset(self, webpage, classification, count):
        self.client.gui.display_message('\nAdding ' + repr(webpage.identifier) + ' to dataset as dataset #' + repr(count + 1))

        # get frequency from webpage
        frequency = webpage.frequency

        # extract the chars and their counts into two lists
        chars, counts = map(list,zip(*frequency))

        sum = 0
        for x in range(len(counts)):
            sum = counts[x] + sum

        for x in range(len(counts)):
            counts[x] = counts[x] / sum

        # insert classification at index 0
        counts.insert(0, classification)
        # insert classification at index 0, pushing
        counts.insert(0, count + 1)


        formattedCounts = ' '.join(str(x) for x in counts)

        # open dataset text file.
        with open('datasets/our_dataset.txt', 'a') as file:
            file.write('\n' + formattedCounts)
