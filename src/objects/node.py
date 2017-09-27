
class Node:
    def __init__(self, identifier):
        super().__init__()
        self.__identifier = identifier
        self.__children = []

    @property
    def identifier(self):
        return self.__identifier

    @property
    def children(self):
        return self.__children

    def add_child(self, webpage):
        self.__children.append(webpage)
