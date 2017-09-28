from cx_Freeze import setup, Executable
import os

base = None



executables = [Executable("__main__.py", base=base)]

packages = ["anytree", "bs4", "requests", "uuid", "matplotlib", "numpy", "tkinter.filedialog", "idna", "io", "re", "base64", "threading", "collections"]
options = {
    'build_exe': {

        'packages': packages,
        "include_files": [r"C:\\Python36\\DLLs\\tcl86t.dll", r"C:\\Python36\\DLLs\\tk86t.dll", "gui.py","src/objects/node.py", "src/objects/web_page.py", "src/objects/tree.py", "src/objects/web_scraper.py", "html_files"]

    },

}

os.environ['TCL_LIBRARY'] = r'C:\\Python36\\tcl\\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Python36\\tcl\\tk8.6'

setup(
    name = "Python Web Scraper",
    options = options,
    version = "1.0.0",
    description = 'python web scraper',
    executables = executables
)
