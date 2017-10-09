import threading


class MachineLearner(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, target=self.run)
        self.__datasets = None

    def run(self):
        print("Machine Learner initialized.")
        self.__datasets = self.extract_datasets()
        print("Index: {}".format(self.datasets[0][0]))
        print("Classification Label: {}".format(self.datasets[0][1]))

    @property
    def datasets(self):
        """ Datasets is a 2-dimension array. datasets[dataset][column]. First column in every dataset is the index, second is classification label, and the rest are the normalized unigram feature vectors. """
        return self.__datasets

    def extract_datasets(self):
        with open ('datasets/our_dataset.txt') as myfile:
            lines = myfile.readlines()
            dataset = []
            for line in lines:
                array = []
                for word in line.split():
                    array.append(word)
                dataset.append(array)
            myfile.close()
            return dataset

        # file = open('./datasets/our_dataset.txt')
        # lines = file.readlines()
        # print(lines)
        # file.close()
