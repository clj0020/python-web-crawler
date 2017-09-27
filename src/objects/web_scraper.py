import threading
import requests
from bs4 import BeautifulSoup
import uuid, base64
import io
from .tree import Tree

class WebScraper(threading.Thread):

    def __init__(self, client, url, depth):
        super().__init__(daemon=True, target=self.run)
        self.client = client
        self.url = url
        self.depth = depth
        self.tree = Tree(self)

    def run(self):
        print("Web Scraper initialized.")
        self.tree.add_node(self.url)
        self.iterative_deepening_search(self.url, self.depth)

    def depth_limited_search(self, node, depth):

        def recursive_depth_limited_search(node, depth):

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
                return result
