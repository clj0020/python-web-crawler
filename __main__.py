"""This is the main entry point for the application."""
import threading
import time
import queue
from gui import *
from src import WebScraper
from src import MachineLearner
import numpy.core._methods
import numpy.lib.format

ENCODING = 'utf-8'

class MainApplication(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, target=self.run)

        self.buffer_size = 1024

        self.queue = queue.Queue()
        self.lock = threading.RLock()

        self.gui = GUI(self)
        self.web_scraper = None
        self.machine_learner = None

        self.start()
        self.gui.start()
        # Only gui is non-daemon thread, therefore after closing gui app will quit


    def run(self):
        """Handle client-server communication using select module"""
        while True:
            if not self.queue.empty():
                data = self.queue.get()
                self.send_message(data)
                self.queue.task_done()
            else:
                time.sleep(0.05)

    def send_message(self, data):
        """"Send encoded message to server"""
        if data:
            message = data.decode(ENCODING)
            message = message.split('\n')
            for msg in message:
                if msg != '':
                    msg = msg.split(';')
                    if msg[0] == 'msg':
                        url = msg[1]
                        print('URL:', url)

                        depth = int(msg[2])
                        print('DEPTH:', depth)

                        self.web_scraper = WebScraper(self, url, depth)
                        self.web_scraper.start()
                    elif msg[0] == 'machine_learner':

                        k = int(msg[1])
                        print('K', k)

                        split = float(msg[2])
                        print('Split', split)

                        self.machine_learner.initialize_k_nearest_neighbor(k, split)


    def create_machine_learner(self):
       self.machine_learner = MachineLearner(self)
       self.machine_learner.start()

    def get_node(self, node):
        return self.web_scraper.tree[node]


if __name__ == "__main__":
    MainApplication()
