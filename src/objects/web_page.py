from .node import Node
import threading
import uuid, base64
import requests
from bs4 import BeautifulSoup
from collections import Counter
from string import printable
from pandas import Series
from pandas import Categorical
import io
import re

class WebPage(Node):

    def __init__(self, client, url):
        super().__init__(url)

        self.__client = client
        # self.__soup = self.scrape_site()
        self.__frequency = None
        self.__tree = None
        # self.__filename = None

        # self.save_file()
        # file_save = threading.Thread(target=self.save_file, daemon=True)
        # file_save.start()

    @property
    def filename(self):
        return self.__filename

    @property
    def tree(self):
        return self.__tree

    @property
    def soup(self):
        return self.__soup

    @property
    def frequency(self):
        return self.__frequency

    @property
    def client(self):
        return self.__client

    def set_tree(self, tree):
        self.__tree = tree

    def create_filename(self):
        uuidstring = str(uuid.uuid5(uuid.NAMESPACE_DNS, self.identifier))
        filename = base64.b64encode(uuid.UUID(uuidstring).bytes).decode("ascii").rstrip('=\n').replace('/', '_')
        return filename

    def scrape_site(self):
        response = requests.get(self.identifier)
        html = response.content
        soup = BeautifulSoup(html, 'html.parser')

        print("Scraped {}...".format(self.identifier))

        return soup

    def find_children(self):
        for link in self.soup.find_all('a', href=True):
            href = link.get('href')
            if (href.startswith('/')):
                href = self.identifier + href
                self.tree.add_node(href, parent=self.identifier)
            elif (href.startswith(self.identifier)):
                continue
            elif (href.startswith('http')):
                self.tree.add_node(href, parent=self.identifier)
        return self.children

    def save_file(self):
        self.__soup = self.scrape_site()

        self.__filename = self.create_filename()

        with io.open('html_files/' + self.filename + '.txt', "wb") as f:
            f.write(self.soup.encode("ascii"))

        print("Saved file for {}".format(self.identifier))

        self.__frequency = self.unigram_extraction()

        self.client.gui.display_message('\nURL: ' + repr(self.identifier))

    def unigram_extraction(self):
        ascii_string = set(""" !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~""")

        file = open('html_files/' + self.filename + '.txt', 'r', encoding="ascii")
        series = Series(Categorical([character for line in file for character in line], categories=ascii_string)).value_counts()
        return sorted(series.items())

        # return sorted(Counter(character for line in file for character in line if character in ascii_string).items())
    #
    # def word_extraction(self):
    #     words = re.findall('\w+', open('html_files/' + self.filename + '.txt').read().lower())
    #     return Counter(words).most_common(30)
