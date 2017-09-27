from .node import Node
import threading
import uuid, base64
import requests
from bs4 import BeautifulSoup
from collections import Counter
import io
import re

class WebPage(Node):

    def __init__(self, url, tree):
        super().__init__(url)

        self.__filename = self.create_filename()
        self.__soup = self.scrape_site()
        self.__frequency = None
        self.__tree = tree

        file_save = threading.Thread(target=self.save_file, daemon=True)

        file_save.start()

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
        with io.open('html_files/' + self.filename + '.txt', "wb") as f:
            f.write(self.soup.encode("ascii"))

        print("Saved file for {}".format(self.identifier))

        self.__frequency = self.unigram_extraction()

        self.tree.scraper.client.gui.display_message(self.identifier)

    def unigram_extraction(self):
        file = open('html_files/' + self.filename + '.txt', 'r', encoding="ascii")
        return sorted(Counter(c for l in file for c in l).items())
    #
    # def word_extraction(self):
    #     words = re.findall('\w+', open('html_files/' + self.filename + '.txt').read().lower())
    #     return Counter(words).most_common(30)
