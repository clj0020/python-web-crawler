import threading
import requests
from bs4 import BeautifulSoup
import uuid, base64
import io
from .tree import Tree
from .webpage_classifier import WebpageClassifier

class WebScraper(threading.Thread):

    def __init__(self, client, url, depth):
        super().__init__(daemon=True, target=self.run)
        self.client = client
        self.url = url
        self.depth = depth
        self.tree = Tree(self)
        self.webpage_classifier = WebpageClassifier(self.client)
        self.websites_added = []

    def run(self):
        print("Web Scraper initialized.")
        self.tree.add_node(self.url)
        self.iterative_deepening_search(self.url, self.depth)

    def depth_limited_search(self, node, depth):

        def recursive_depth_limited_search(node, depth):

            file_save = threading.Thread(target=self.tree[node].save_file, daemon=True)
            file_save.start()
            file_save.join()


            unigram_vector = self.webpage_classifier.scrape_site(self.tree[node])
            positives, negatives = self.webpage_classifier.balance_dataset(self.webpage_classifier.dataset)
            self.client.gui.display_message('\nThere are currently ' + repr(positives) + 'malicious sites and ' + repr(negatives) + 'trusted sites in the dataset.')
            if unigram_vector[1] == -1.0:
                if positives > negatives:
                    formatted_unigram_vector = ' '.join(str(x) for x in unigram_vector)
                    # open dataset text file.
                    with open('datasets/our_dataset.txt', 'a') as file:
                        file.write(formatted_unigram_vector + '\n')
                        file.close()
                    self.webpage_classifier.dataset.append(unigram_vector)
                    self.websites_added.append(self.tree[node].identifier)
                    self.client.gui.display_message('\nAdded ' + repr(self.tree[node].identifier) + ' to dataset as dataset #' + repr(unigram_vector[0]))

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
