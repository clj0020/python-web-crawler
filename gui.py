import tkinter as tk
import threading
from tkinter import scrolledtext
from tkinter import messagebox

import matplotlib.pyplot as plt
import numpy as np

ENCODING = 'utf-8'

class GUI(threading.Thread):
    def __init__(self, client):
        super().__init__(daemon=False, target=self.run)
        self.font = ('Helvetica', 13)
        self.client = client
        self.main_window = None

    def run(self):
        self.main_window = MainWindow(self, self.font)
        self.main_window.run()

    @staticmethod
    def display_alert(message):
        """Display alert box"""
        messagebox.showinfo('Error', message)

    def display_message(self, message):
        """Display message in MainWindow"""
        self.main_window.display_message(message)

    def send_message(self, message):
        """Enqueue message in client's queue"""
        self.client.queue.put(message)

class Window(object):
    def __init__(self, title, font):
        self.root = tk.Tk()
        self.title = title
        self.root.title(title)
        self.font = font

class MainWindow(Window):
    def __init__(self, gui, font):
        super().__init__("Python Web Crawler", font)
        self.gui = gui
        self.lock = threading.RLock()
        self.target = ''
        self.url_entry = None
        self.depth_entry = None
        self.submit_button = None
        self.url_list = None
        self.unigram_table = None
        self.target = None
        self.webpage_frame = None
        self.url_display = None

        self.build_window()

    def build_window(self):
        """Build main window, set widgets positioning and event bindings"""

        form_frame = tk.Frame(self.root)
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
        self.submit_button.grid(row=2, column=0, columnspan=2, sticky='W')

        url_list_frame = tk.Frame(self.root)
        url_list_frame.pack(fill='x')

        self.url_list = tk.Listbox(url_list_frame, selectmode=tk.SINGLE, font=self.font,
                                      exportselection=False)
        self.url_list.bind('<<ListboxSelect>>', self.selected_url_event)
        self.url_list.pack(fill=tk.BOTH, expand=tk.YES)

        self.webpage_frame = tk.Frame(self.root)
        self.webpage_frame.pack()

        self.url_display = tk.Label(self.webpage_frame, text = "", width = 15)
        self.url_display.pack()

        # Protocol for closing window using 'x' button
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing_event)

    def run(self):
        """Handle chat window actions"""
        self.root.mainloop()
        self.root.destroy()

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

    def exit_event(self, event):
        """Quit app when "Exit" pressed"""
        # self.gui.notify_server(self.login, 'logout')
        self.root.quit()

    def on_closing_event(self):
        """Exit window when 'x' button is pressed"""
        self.exit_event(None)

    def display_message(self, message):
        """Display message in ScrolledText widget"""
        with self.lock:
            self.url_list.configure(state='normal')
            self.url_list.insert(tk.END, message)
            self.url_list.see(tk.END)
