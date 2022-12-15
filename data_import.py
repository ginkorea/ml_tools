"""Module is used to get input for arrays and input from the user"""
import os
import pandas as pd
import instructions as inst
import manipulation as mn


def process(k):
    """function to load the data and process it, takes k number of folds as input"""
    crawler = Crawler()  # creates a crawler to search the data folder and import data
    database = []  # holds the different manipulated datasets
    for current in crawler.datasets:
        cv = mn.CrossValidator(current, k)  # creates a cross-validator with k folds
        if current.operation == 0:  # classification
            print("conducting classification on %s" % current.title)
            cv.classify()
        elif current.operation == 1:  # regression
            print("conducting regression on %s" % current.title)
            cv.regression()
        database.append(cv)
    return database


class Data:
    """class used to hold the data"""
    source = None
    names = None
    title = None
    operation = None
    means = []
    hyper_tuned = False
    records = 0

    def __init__(self, source):
        self.source = source
        self.matrix = []
        self.import_data()
        self.title = self.matrix.pop(0)  # row 0 used for title
        self.title = self.title[0]
        self.names = self.matrix.pop(0)  # row 1 for the names of columns
        self.instructions = self.matrix.pop(0)  # [number,classifier,{args}] used to provide handeling instructions
        if int(self.instructions[0]) in [0, 1, 2, 20]:
            self.operation = 0  # classification
        elif int(self.instructions[0]) in [3, 4, 5, 21]:
            self.operation = 1  # regression
        self.columns = len(self.names)
        self.normalize_data()
        self.df = pd.DataFrame(data=self.matrix, columns=self.names)
        print(self.df)

    def import_data(self):
        """reads data from a csv"""
        try:
            with open(self.source, 'r') as data:  # read text file
                print(data.name)
                for record in data:  # iterate through all lines in the file
                    record = record[:-1]
                    record.strip("\n")
                    record = record.split(",")
                    try:  # error handling for data type and input errors
                        self.matrix.append(record)  # add point to the plane
                    except IndexError:  # for blank lines
                        continue  # skip line
                self.records = len(self.matrix)
                print("Imported {0} rows from {1}".format(self.records - 3, self.source))
        except IOError:  # if there is no .txt file
            print("Unable to import data.")
        data.close()

    def normalize_data(self):
        """set of instructions to handle the normalization of data"""
        instruction, classifier = inst.get_instructions(self)
        if instruction in [0]:  # binary data with missing rows
            inst.fill_missing_data(self)
            inst.normalize_binary_classifier(self)
        elif instruction in [1]:  # ordinal classification
            inst.normalize_ordinal_classifier(self)
        elif instruction in [2]:  # nominal classification
            inst.normalize_nominal_classifier(self)
        elif instruction in [4, 5]:  # unneeded columns regression
            inst.discard_columns(self)


class Crawler:
    """class used to search data directory and import data"""

    def __init__(self):
        self.datasets = []
        file_path = os.path.realpath(__file__)  # get the path for where the program is being executed
        self.root = os.path.dirname(file_path)  # get the directory for the application
        self.data = self.root + "\\data"
        self.get_data_files()
        self.create_matrix()

    def get_data_files(self):
        """used to append the program directory path to the requested filename"""
        data_files = os.listdir(self.data)
        for dataset in data_files:
            data_path = self.data + "\\" + dataset
            self.datasets.append(data_path)

    def create_matrix(self):
        """creates the matrix and replaces the source dataset with a Data object"""
        replacement = []
        for dataset in self.datasets:
            d = Data(dataset)
            replacement.append(d)
        self.datasets = replacement
