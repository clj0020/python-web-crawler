import threading
import math
import xml.etree.ElementTree as ET
import urllib.request
from .web_page import WebPage
from .distance_weighted_k_nearest_neighbor import DistanceWeightedKNearestNeighbor

class WebpageClassifier(threading.Thread):

    def __init__(self, client):
        super().__init__(daemon=True, target=self.run)
        self.__client = client
        self.__dataset = self.extract_dataset()

    def run(self):
        print("Webpage Classifier initialized...")

    @property
    def client(self):
        return self.__client

    @property
    def dataset(self):
        return self.__dataset

    def extract_dataset(self):
        dataset = []
        if self.client is not None:
            self.client.gui.display_message("\nLoading dataset...")
        with open('datasets/our_dataset.txt') as myfile:
            lines = myfile.readlines()

            print("Number of Datasets: " + repr(len(lines)))

            # Create a 2d array with the numbers in the dataset file
            for line in lines:
                array = []
                for number in line.split():
                    array.append(float(number))
                dataset.append(array)

            myfile.close()
        self.client.gui.display_message("\nDataset loaded!")
        return dataset

    def scrape_site(self, webpage):
        # print("Scraping site: {}".format(url))
        # webpage = WebPage(self.client, url)
        frequency = webpage.frequency

        print(len(frequency))

        # extract the chars and their unigram_vector into two lists
        chars, unigram_vector = map(list,zip(*frequency))

        sum = 0
        for x in range(len(unigram_vector)):
            sum = unigram_vector[x] + sum

        for x in range(len(unigram_vector)):
            unigram_vector[x] = unigram_vector[x] / sum

        self.normalize_dataset(unigram_vector)

        classification = 0 # temp

        unigram_vector.insert(0, classification)
        unigram_vector.insert(0, len(self.dataset))

        distance_weighted_k_nearest_neighbor = DistanceWeightedKNearestNeighbor(self.client, 6, False)
        distance_weighted_k_nearest_neighbor.start()
        distance_weighted_k_nearest_neighbor.join()
        distance_weighted_k_nearest_neighbor.distance_weighted_k_nearest_neighbor_single(6, unigram_vector)

        actual = self.check_classification(webpage.identifier)

        self.client.gui.display_message("\nActual: " + repr(actual))
        unigram_vector[1] = actual

        return unigram_vector

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

        # extract the chars and their unigram_vector into two lists
        chars, unigram_vector = map(list,zip(*frequency))

        sum = 0
        for x in range(len(unigram_vector)):
            sum = unigram_vector[x] + sum

        for x in range(len(unigram_vector)):
            unigram_vector[x] = unigram_vector[x] / sum

        # insert classification at index 0
        unigram_vector.insert(0, classification)
        # insert classification at index 0, pushing
        unigram_vector.insert(0, count + 1)


        formatted_unigram_vector = ' '.join(str(x) for x in unigram_vector)

        # open dataset text file.
        with open('datasets/our_dataset.txt', 'a') as file:
            file.write('\n' + formatted_unigram_vector)

    def check_classification(self, website):
        query = 'http://api.urlvoid.com/api1000/e0b84941aec0e9d9b7baea4da1741d4cae29b7e4/host/'

        website = website.replace("http://","")
        website = website.replace("https://","")
        website = website.replace("www.", "") # May replace some false positives ('www.com')

        fullQuery = query + website + '/'
        #sending a request to the API
        with urllib.request.urlopen(fullQuery) as response:
            #reading the request to an object
            xmlData = response.read()
        root = ET.fromstring(xmlData)

        if root.find('details') is None:
            return 0

        rank = -1
        for child in root.findall('detections'):
            rank = float(child.find('count').text)
        return rank

    def balance_dataset(self, dataset):
        positives = 0
        negatives = 0
        for x in range(len(dataset)):
            classification = dataset[x][1]
            if classification == 1:
                positives = positives + 1
            elif classification == -1:
                negatives = negatives + 1

        print("Positives: " + repr(positives))
        print("Negatives: " + repr(negatives))
        return positives, negatives

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
