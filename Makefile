# the executable for pip
PIP=pip
# the executable for Python
PYTHON=python

# run the command to install dependencies in the requirements.txt file.
install:
	${PIP} install -r requirements.txt

# run the command to execute the unit tests
test:
	${PYTHON} -m unittest discover src

# run the command to start the application
start:
	${PYTHON} .

# build executable
executable:
	${PYTHON} setup.py build
