# from .node import Node
from .web_page import WebPage

(_ROOT, _DEPTH, _BREADTH) = range(3)

class Tree:

    def __init__(self, scraper):
        super().__init__()
        self.__nodes = {}
        self.__scraper = scraper

    @property
    def nodes(self):
        return self.__nodes

    @property
    def scraper(self):
        return self.__scraper

    def add_node(self, identifier, parent=None):
        webpage = WebPage(self.scraper.client, identifier)
        webpage.set_tree(self)
        self[identifier] = webpage

        if parent is not None:
            self[parent].add_child(identifier)

        return webpage

    def __getitem__(self, key):
        return self.__nodes[key]

    def __setitem__(self, key, item):
        self.__nodes[key] = item
