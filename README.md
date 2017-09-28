# Python Application Boilerplate

This is a web crawler built in python that implements Iterative Deepening Depth Search to scrape all of the children links of a specified base url up to a specified depth. While scraping, the program saves each page's HTML into a text file and the runs a Unigram Feature Extractor on those files. Once dependencies are installed and the program is run, simply enter an url and the depth that you want to search and press submit. The program will then start scraping sites, and once the site is saved to the html_files folder, it will be added to a list in the user interface that allows you to view that sites unigram features as a graph when you click on them.

Note: Written in Python 3

# Usage

## Installing dependencies

To install dependency Python modules in the [requirements.txt][] file:

```shell
make install
```

[Makefile]: ./Makefile
[requirements.txt]: ./requirements.txt

## Running tests

To run the test modules inside the [src][] package:

```shell
make test
```

[src]: ./src

## Starting the application

To execute the code in the [__main__.py][] script:

```shell
make start
```

[__main__.py]: ./__main__.py

## Creating an .exe file (Windows)

To create an .exe file in the build package:

```shell
make executable
```

[Makefile]: ./Makefile
[requirements.txt]: ./requirements.txt


# Makefile

[Makefile][] contains all the shortcuts for frequently used project commands.

Note: make sure you have [Make](https://www.gnu.org/software/make/) installed.

## Python3

If you have python and run into issues, make sure you're shell is pointing to the right version of Python with `python --version`.
