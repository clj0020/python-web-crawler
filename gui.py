import tkinter as tk
import threading
from tkinter import *
from tkinter import scrolledtext

import matplotlib.pyplot as plt
import numpy as np

ENCODING = 'utf-8'

class GUI(threading.Thread):
    def __init__(self, client):
        super().__init__(daemon=False, target=self.run)
        self.font = ('Helvetica', 13)
        self.client = client
        self.main_window = None
        self.machine_learner_window = None
        self.web_crawler_window = None
        self.webpage_classifier_window = None

    def run(self):
        self.main_window = MainWindow(self, self.font)
        self.main_window.run()

    @staticmethod
    def display_alert(message):
        """Display alert box"""
        messagebox.showinfo('Error', message)

    def display_message(self, message):
        """Display message in Open Window"""
        if self.web_crawler_window is None and self.webpage_classifier_window is None:
            self.machine_learner_window.display_message(message)
        elif self.web_crawler_window is None and self.machine_learner_window is None:
            self.webpage_classifier_window.display_message(message)
        elif self.webpage_classifier_window is None and self.machine_learner_window is None:
            self.web_crawler_window.display_message(message)

    def send_message(self, message):
        """Enqueue message in client's queue"""
        self.client.queue.put(message)

    def open_machine_learner_window(self, root):
        self.machine_learner_window = MachineLearnerWindow(self, self.font, root)
        self.client.create_machine_learner()
        self.machine_learner_window.run()

    def open_web_crawler_window(self, root):
        self.web_crawler_window = WebCrawlerWindow(self, self.font, root)
        self.web_crawler_window.run()

    def open_webpage_classifier_window(self, root):
        self.webpage_classifier_window = WebPageClassifierWindow(self, self.font, root)
        self.client.create_webpage_classifer()
        self.webpage_classifier_window.run()

class Window(object):
    def __init__(self, title, font):
        self.root = tk.Tk()
        self.title = title
        self.root.title(title)
        self.font = font

class MainWindow(Window):
    def __init__(self, gui, font):
        super().__init__("Computational Intelligence Express", font)
        self.gui = gui
        self.lock = threading.RLock()
        self.open_machine_learner_window_button = None
        self.open_web_crawler_window_button = None
        self.open_webpage_classifier_window_button = None

        self.build_window()
        self.run()

    def build_window(self):
        """Build main window, set widgets positioning and event bindings"""

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both')

        self.open_machine_learner_window_button = tk.Button(main_frame, text="Open Machine Learner")
        self.open_machine_learner_window_button.bind('<Button-1>', self.open_machine_learner_window)
        self.open_machine_learner_window_button.pack(side="left")

        self.open_web_crawler_window_button = tk.Button(main_frame, text="Open Web Crawler")
        self.open_web_crawler_window_button.bind('<Button-1>', self.open_web_crawler_window)
        self.open_web_crawler_window_button.pack(side="left")

        self.open_webpage_classifier_window_button = tk.Button(main_frame, text="Open WebPage Classifier")
        self.open_webpage_classifier_window_button.bind('<Button-1>', self.open_webpage_classifier_window)
        self.open_webpage_classifier_window_button.pack(side="left")

        # Protocol for closing window using 'x' button
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing_event)

    def run(self):
        self.root.mainloop()

    def exit_event(self, event):
        """Quit app when "Exit" pressed"""
        self.root.quit()

    def on_closing_event(self):
        """Exit window when 'x' button is pressed"""
        self.exit_event(None)

    def open_machine_learner_window(self, event):
        """Open the Machine Learner Window"""
        self.gui.open_machine_learner_window(self.root)

    def open_web_crawler_window(self, event):
        """Open the Web Crawler Window"""
        self.gui.open_web_crawler_window(self.root)

    def open_webpage_classifier_window(self, event):
        """Open the Webpage Classifier Window"""
        self.gui.open_webpage_classifier_window(self.root)


class WebCrawlerWindow():
    def __init__(self, gui, font, root):
        self.root = root
        self.window = tk.Toplevel(self.root)
        self.gui = gui
        self.font = font
        self.lock = threading.RLock()
        self.target = ''
        self.url_entry = None
        self.depth_entry = None
        self.submit_button = None
        self.url_list = None
        self.unigram_table = None
        self.target = None
        self.open_machine_learner_window_button = None

        self.build_window()

    def build_window(self):
        """Build main window, set widgets positioning and event bindings"""

        form_frame = tk.Frame(self.window)
        form_frame.pack(fill='x')

        tk.Label(form_frame, text="Url").grid(row=0, column=0, sticky='W')
        url = tk.StringVar()
        self.url_entry = tk.Entry(form_frame, textvariable=url)
        self.url_entry.grid(row=0, column=1, sticky='W')

        tk.Label(form_frame, text="Depth").grid(row=1, column=0, sticky='W')
        depth = tk.IntVar()
        self.depth_entry = tk.Entry(form_frame, textvariable=depth)
        self.depth_entry.grid(row=1, column=1, sticky='W')

        self.submit_button = tk.Button(form_frame, text="Submit")
        self.submit_button.bind('<Button-1>', self.send_entry_event)
        self.submit_button.grid(row=2, column=0, columnspan=1, sticky='W')

        url_list_frame = tk.Frame(self.window)
        url_list_frame.pack(fill='x')

        self.url_list = tk.Listbox(url_list_frame, selectmode=tk.SINGLE, font=self.font,
                                      exportselection=False)
        self.url_list.bind('<<ListboxSelect>>', self.selected_url_event)
        self.url_list.pack(fill=tk.BOTH, expand=tk.YES)

    def run(self):
        """Handle chat window actions"""
        # self.root.mainloop()
        # self.root.destroy()

    def selected_url_event(self, event):
        """Set as target currently selected login on login list"""
        target = self.url_list.get(self.url_list.curselection())
        current_webpage = self.gui.client.get_node(target)
        frequency = current_webpage.frequency

        chars, counts = map(list,zip(*frequency))
        y_pos = np.arange(len(chars))
        plt.bar(y_pos, counts, align='center', alpha=0.5)
        plt.xticks(y_pos, chars)
        plt.ylabel('Frequency')
        plt.title('Unigram Features')

        plt.show()

    def send_entry_event(self, event):
        """Send message from entry field to target"""
        url = self.url_entry.get()
        depth = self.depth_entry.get()

        print(url)
        print(depth)

        if url != '\n' and depth != 0:
            message = 'msg;' + url + ';' + depth
            print(message)
            self.gui.send_message(message.encode(ENCODING))
        else:
            messagebox.showinfo('Warning', 'You must enter valid url and depth.')
        return 'break'

    def display_message(self, message):
        """Display message in ScrolledText widget"""
        with self.lock:
            self.url_list.configure(state='normal')
            self.url_list.insert(tk.END, message)
            self.url_list.see(tk.END)

class MachineLearnerWindow():
    def __init__(self, gui, font, root):
        self.root = root
        self.window = tk.Toplevel(self.root)
        self.font = font
        self.gui = gui
        self.lock = threading.RLock()
        self.k_entry = None
        self.k_nearest_neighbor_button = None
        self.k_nearest_test_button = None

        self.weighted_k_entry = None
        self.weighted_option = None
        self.weighted_global_checkbox = None
        self.weighted_k_nearest_neighbor_button = None
        self.weighted_k_nearest_test_button = None

        self.general_regression_neural_network_button = None
        self.messages_list = None

        self.build_window()

    def build_window(self):
        """Build main window, set widgets positioning and event bindings"""
        # Size config
        self.window.geometry('750x500')
        self.window.minsize(600, 400)

        main_frame = tk.Frame(self.window)
        main_frame.pack(fill="both")

        top_frame = tk.Frame(main_frame)
        top_frame.pack(side="top", fill="x")

        bottom_frame = tk.Frame(main_frame)
        bottom_frame.pack(side="bottom", fill="x")

        k_nearest_neighbor_frame = tk.Frame(top_frame)
        k_nearest_neighbor_frame.pack(side="left")

        tk.Label(k_nearest_neighbor_frame, text="K-Nearest Neighbor").pack(side="top")

        k_nearest_neighbor_form_frame = tk.Frame(k_nearest_neighbor_frame)
        k_nearest_neighbor_form_frame.pack(side="top")

        k_nearest_neighbor_k_entry_frame = tk.Frame(k_nearest_neighbor_form_frame)
        k_nearest_neighbor_k_entry_frame.pack(side="top")

        tk.Label(k_nearest_neighbor_k_entry_frame, text="K").pack(side="left")
        k = tk.IntVar()
        self.k_entry = tk.Entry(k_nearest_neighbor_k_entry_frame, textvariable=k)
        self.k_entry.pack(side="right")

        self.k_nearest_test_button = tk.Button(k_nearest_neighbor_form_frame, text="Find Best K-Value (Will Take Time)")
        self.k_nearest_test_button.bind('<Button-1>', self.start_k_nearest_neighbor_evaluation)
        self.k_nearest_test_button.pack(side="bottom")

        self.k_nearest_neighbor_button = tk.Button(k_nearest_neighbor_form_frame, text="Submit")
        self.k_nearest_neighbor_button.bind('<Button-1>', self.start_k_nearest_neighbor)
        self.k_nearest_neighbor_button.pack(side="bottom")


        weighted_k_nearest_neighbor_frame = tk.Frame(top_frame)
        weighted_k_nearest_neighbor_frame.pack(side="left")

        tk.Label(weighted_k_nearest_neighbor_frame, text="Distance Weighted K-Nearest Neighbor").pack(side="top")

        weighted_k_nearest_neighbor_form_frame = tk.Frame(weighted_k_nearest_neighbor_frame)
        weighted_k_nearest_neighbor_form_frame.pack(side="top")

        weighted_k_nearest_neighbor_k_entry_frame = tk.Frame(weighted_k_nearest_neighbor_form_frame)
        weighted_k_nearest_neighbor_k_entry_frame.pack(side="top")

        tk.Label(weighted_k_nearest_neighbor_k_entry_frame, text="K").pack(side="left")
        k = tk.IntVar()
        self.weighted_k_entry = tk.Entry(weighted_k_nearest_neighbor_k_entry_frame, textvariable=k)
        self.weighted_k_entry.pack(side="right")

        weighted_k_nearest_neighbor_options_frame = tk.Frame(weighted_k_nearest_neighbor_form_frame)
        weighted_k_nearest_neighbor_options_frame.pack(side="top")

        self.weighted_option = tk.BooleanVar()
        self.weighted_global_checkbox = tk.Checkbutton(weighted_k_nearest_neighbor_options_frame, text="Global", variable=self.weighted_option)
        self.weighted_global_checkbox.pack()

        self.weighted_k_nearest_test_button = tk.Button(weighted_k_nearest_neighbor_form_frame, text="Find Best K-Value (Will Take Time)")
        self.weighted_k_nearest_test_button.bind('<Button-1>', self.start_distance_k_nearest_neighbor_evaluation)
        self.weighted_k_nearest_test_button.pack(side="bottom")

        self.weighted_k_nearest_neighbor_button = tk.Button(weighted_k_nearest_neighbor_form_frame, text="Submit")
        self.weighted_k_nearest_neighbor_button.bind('<Button-1>', self.start_distance_weighted_k_nearest_neighbor)
        self.weighted_k_nearest_neighbor_button.pack(side="bottom")

        general_regression_neural_network_frame = tk.Frame(top_frame)
        general_regression_neural_network_frame.pack(side="left")

        tk.Label(general_regression_neural_network_frame, text="General Regression Neural Network").pack(side="top")

        self.general_regression_neural_network_button = tk.Button(general_regression_neural_network_frame, text="Submit")
        self.general_regression_neural_network_button.bind('<Button-1>', self.start_general_regression_neural_network)
        self.general_regression_neural_network_button.pack()

        # ScrolledText widget for displaying messages
        self.messages_list = scrolledtext.ScrolledText(bottom_frame, wrap='word', font=self.font)
        self.messages_list.configure(state='disabled')
        self.messages_list.pack(fill="x")

        # # Protocol for closing window using 'x' button
        # self.root.protocol("WM_DELETE_WINDOW", self.on_closing_event)

    def start_k_nearest_neighbor(self, event):
        k = self.k_entry.get()

        if k != 0:
            message = 'machine_learner;' + 'k-nearest;' + k
            print(message)
            self.gui.send_message(message.encode(ENCODING))
        else:
            messagebox.showinfo('Warning', 'You must enter valid k.')
        return 'break'

    def start_k_nearest_neighbor_evaluation(self, event):
        message = 'machine_learner;' + 'evaluate_k_nearest;'
        self.gui.send_message(message.encode(ENCODING))
        return 'break'

    def start_distance_weighted_k_nearest_neighbor(self, event):
        k = self.weighted_k_entry.get()
        is_global = self.weighted_option.get()

        if k != 0:
            message = 'machine_learner;' + 'distance-weighted;' + k + ';' + str(is_global)
            print(message)
            self.gui.send_message(message.encode(ENCODING))
        else:
            messagebox.showinfo('Warning', 'You must enter valid k.')
        return 'break'

    def start_distance_k_nearest_neighbor_evaluation(self, event):
        is_global = self.weighted_option.get()

        message = 'machine_learner;' + 'evaluate_distance_k_nearest;' + str(is_global)
        self.gui.send_message(message.encode(ENCODING))
        return 'break'

    def start_general_regression_neural_network(self, event):
        message = 'machine_learner;' + 'grnn;'
        self.gui.send_message(message.encode(ENCODING))
        return 'break'

    def display_message(self, message):
        """Display message in ScrolledText widget"""
        with self.lock:
            self.messages_list.configure(state='normal')
            self.messages_list.insert(tk.END, message)
            self.messages_list.configure(state='disabled')
            self.messages_list.see(tk.END)

    def display_k_nearest_graph(self, accuracies):
        with self.lock:
            k, accuracy = [x[0] for x in accuracies],[x[1] for x in accuracies]
            y_pos = np.arange(len(k))

            plt.title('Accuracy of Different K-Values (KNN)')
            plt.ylabel('Accuracy')
            plt.xlabel('K')

            plt.bar(y_pos, accuracy, align='center', alpha=0.5)

            # You can specify a rotation for the tick labels in degrees or with keywords.
            plt.xticks(y_pos, k)
            # Pad margins so that markers don't get clipped by the axes
            plt.margins(0.2)
            # Tweak spacing to prevent clipping of tick-labels
            plt.subplots_adjust(bottom=0.15)

            plt.savefig('k_nearest_accuracy_graph.png')

    def display_distance_k_nearest_graph(self, accuracies):
        with self.lock:
            k, accuracy = [x[0] for x in accuracies],[x[1] for x in accuracies]
            y_pos = np.arange(len(k))

            plt.title('Accuracy of Different K-Values (Distance Weighted)')
            plt.ylabel('Accuracy')
            plt.xlabel('K')

            plt.bar(y_pos, accuracy, align='center', alpha=0.5)

            # You can specify a rotation for the tick labels in degrees or with keywords.
            plt.xticks(y_pos, k)
            # Pad margins so that markers don't get clipped by the axes
            plt.margins(0.2)
            # Tweak spacing to prevent clipping of tick-labels
            plt.subplots_adjust(bottom=0.15)

            plt.savefig('distance_weighted_k_nearest_accuracy_graph.png')

    def run(self):
        """Handle chat window actions"""

class WebPageClassifierWindow():

    def __init__(self, gui, font, root):
        self.root = root
        self.window = tk.Toplevel(self.root)
        self.font = font
        self.gui = gui
        self.lock = threading.RLock()
        self.url_entry = None
        self.submit_button = None
        self.add_webpages_to_dataset_button = None
        self.messages_list = None

        self.build_window()

    def build_window(self):
        """Build window, set widgets positioning and event bindings"""
        # Size config
        self.window.geometry('750x500')
        self.window.minsize(600, 400)

        main_frame = tk.Frame(self.window)
        main_frame.pack(fill="both")

        top_frame = tk.Frame(main_frame)
        top_frame.pack(side="top", fill="x")

        tk.Label(top_frame, text="Enter a URL to classify").pack(side="top")

        webpage_classifier_form_frame = tk.Frame(top_frame)
        webpage_classifier_form_frame.pack(side="top")

        tk.Label(webpage_classifier_form_frame, text="URL").pack(side="left")
        url = tk.StringVar()
        self.url_entry = tk.Entry(webpage_classifier_form_frame, textvariable=url)
        self.url_entry.pack(side="right")

        self.submit_button = tk.Button(top_frame, text="Scrape Site")
        self.submit_button.bind('<Button-1>', self.scrape_site)
        self.submit_button.pack(side="bottom")

        self.add_webpages_to_dataset_button = tk.Button(top_frame, text="Add Webpages to Dataset")
        self.add_webpages_to_dataset_button.bind('<Button-1>', self.add_webpages_to_dataset)
        self.add_webpages_to_dataset_button.pack(side="bottom")

        bottom_frame = tk.Frame(main_frame)
        bottom_frame.pack(side="bottom", fill="x")

        # ScrolledText widget for displaying messages
        self.messages_list = scrolledtext.ScrolledText(bottom_frame, wrap='word', font=self.font)
        self.messages_list.configure(state='disabled')
        self.messages_list.pack(fill="x")


    def scrape_site(self, event):
        url = self.url_entry.get()

        if url != '\n':
            message = 'webpage_classifier;' + 'scrape_site;' + url
            print(message)
            self.gui.send_message(message.encode(ENCODING))
        else:
            messagebox.showinfo('Warning', 'You must enter valid url.')
        return 'break'

    def add_webpages_to_dataset(self, event):
        message = 'webpage_classifier;' + 'add_webpages;'
        self.gui.send_message(message.encode(ENCODING))


    def display_message(self, message):
        """Display message in ScrolledText widget"""
        with self.lock:
            self.messages_list.configure(state='normal')
            self.messages_list.insert(tk.END, message)
            self.messages_list.configure(state='disabled')
            self.messages_list.see(tk.END)


    def run(self):
        """Web page classifier window actions"""
