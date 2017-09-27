# Python Application Boilerplate

Boilerplate code for building applications in Python.


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

# Makfefile 

[Makefile][] contains all the shortcuts for frequently used project commands.

Note: make sure you have [Make](https://www.gnu.org/software/make/) installed.

## Python3

If you code in Python3 and run into issues, make sure you're shell is pointing to the right version of Python with `python --version`. If it's pointing to something less than 3 you can change the makefile to use `python3` and `pip3`.
