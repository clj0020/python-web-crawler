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

        # start the application
        self.start()
        # start the gui
        self.gui.start()
        # Only gui is non-daemon thread, therefore after closing gui app will quit


    def run(self):
        """Handle client-server communication using select module"""
        # constantly check for messages in the queue and handle them.
        # a forever loop
        while True:
            # if there is something in the queue
            if not self.queue.empty():
                # get the data from the queue
                data = self.queue.get()
                # call the send message method
                self.send_message(data)
                # After send message method is done tell the queue that its done
                self.queue.task_done()
            else:
                # if the queue isn't empty, wait 5 milliseconds.
                time.sleep(0.05)

    # Allows for communication between the GUI and the rest of the classes
    # GUI usually calls this in response to user events.
    def send_message(self, data):
        """"Send encoded message to server"""
        if data:
            message = data.decode(ENCODING)
            message = message.split('\n')
            for msg in message:
                if msg != '':
                    # message consists of variables separated by semi-colons
                    msg = msg.split(';')
                    # if the message is for the webscraper part of the application
                    if msg[0] == 'msg':
                        # get the url and depth from the message
                        url = msg[1]
                        depth = int(msg[2])
                        # initialize the webscraper and start it
                        self.web_scraper = WebScraper(self, url, depth)
                        self.web_scraper.start()
                    # if the message is for the machine learning part of the application
                    elif msg[0] == 'machine_learner':
                        # if the message is for k-nearest
                        if msg[1] == 'k-nearest':
                            # get k and start the k nearest neighbor training
                            k = int(msg[2])
                            self.machine_learner.initialize_k_nearest_neighbor(k)
                        # if the message is for distance weighted k-nearest
                        elif msg[1] == 'distance-weighted':
                            # extract k and the global boolean value from the message
                            k = int(msg[2])
                            if msg[3] == 'True':
                                is_global = True
                            elif msg[3] == 'False':
                                is_global = False
                            # Start the distance weighted knn algorithm.
                            self.machine_learner.initialize_distance_weighted_k_nearest_neighbor(k, is_global)

    # Initialize and create the machine learner object
    def create_machine_learner(self):
       self.machine_learner = MachineLearner(self)
       self.machine_learner.start()

    # pull a node from the tree
    def get_node(self, node):
        return self.web_scraper.tree[node]

# Start the application
if __name__ == "__main__":
    MainApplication()
