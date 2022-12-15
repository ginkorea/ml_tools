import pandas as pd
import numpy as np
import os


def process(k):
    """function to load the data and process it, takes k number of folds as input"""
    crawler = Crawler()  # creates a crawler to search the data folder and import data
    database = []  # holds the different manipulated datasets
    for current in crawler.datasets:
        cv = CrossValidator(current, k)  # creates a cross-validator with k folds
        if current.operation == 0:  # classification
            print("conducting classification on %s" % current.title)
            # cv.classify()
        elif current.operation == 1:  # regression
            print("conducting regression on %s" % current.title)
            # cv.regression()
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
        self.instructions = self.matrix.pop(0)  # [number,classifier,{args}] used to provide handling instructions
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
        instruction, classifier = self.get_instructions()
        if instruction in [0]:  # binary data with missing rows
            self.fill_missing_data()
            self.normalize_binary_classifier()
            self.z_standardize_dataset()
        elif instruction in [1]:  # ordinal classification
            print("do nothing")
            # self.normalize_ordinal_classifier()
        elif instruction in [2]:  # nominal classification
            print("do nothing")
            # self.normalize_nominal_classifier()
        elif instruction in [4, 5]:  # unneeded columns regression
            print("do nothing")
            # self.discard_columns()

    def fill_missing_data(self):
        """fills in the missing data in a matrix"""
        c = 1
        self.means.append(None)
        while c < self.columns:
            value = 0
            e = 0
            replace = []
            for i, r in enumerate(self.matrix):
                try:
                    value += int(r[c])
                except ValueError:
                    e += 1
                    replace.append(i)
            value = float(value / (len(self.matrix) - e))
            self.means.append(value)
            if e > 0:
                for r in replace:
                    self.matrix[r][c] = self.means[c]
                print("replaced %s values" % len(replace))
            c += 1

    def normalize_binary_classifier(self):
        """function to normalize binary data i.e. benign or malignant tumors"""
        print("normalizing binary data")
        classifier = int(self.instructions[1])
        false = self.instructions[2]
        true = self.instructions[3]
        for row in self.matrix:
            if row[classifier] == false:
                row[classifier] = 0
            elif row[classifier] == true:
                row[classifier] = 1

    def discard_columns(self):
        """function for discarding columns at the beginning of a matrix"""
        print("discarding columns")
        start = int(self.instructions[2])  # hard coded location for column to start on
        for i, m in enumerate(self.matrix):
            self.matrix[i] = m[start:]  # take slice of matrix

    def get_instructions(self):
        """function for getting the instructions for processing the dataset"""
        numb, classifier = int(self.instructions[0]), int(
            self.instructions[1])  # position 0 is handling instructions
        # position 1 is for the row of importance (i.e. classification or regression)
        return numb, classifier

    def convert_column_to_float(self, column):
        """function to convert data in a column into a float"""
        array = []
        for row in self.matrix:  # iterates through matrix and replaces it with float values
            try:
                row[column] = float(row[column])
                array.append(row)
            except TypeError:
                print("Type Error: unable to parse %s to float" % row[column])
        return array  # returns column

    def get_column(self, column):
        this_column = []
        for row in self.matrix:
            this_column.append(float(row[column]))
        this_array = np.array(this_column)
        return this_array

    def z_standardize_column(self, column):
        """function to perform z-standardization on a column in a matrix"""
        column_array = self.get_column(column)
        sd = np.std(column_array)
        mean = np.mean(column_array)
        for i, x in enumerate(column_array):
            z = z_standardize(sd, mean, x)
            self.matrix[i][column] = z
            column_array[i] = z
        return column_array

    def z_standardize_dataset(self, exclude=None):
        if exclude is None:
            exclude = [0, 10]
        i = 0
        while i < self.columns:
            if i not in exclude:
                self.z_standardize_column(i)
            i += 1

    def split_by_class_binary(self):
        numb, column = self.get_instructions()
        by_class = [[], []]
        for row in self.matrix:
            this_record = row[column]
            by_class[this_record].append(row)
        for i, this_class in enumerate(by_class):
            randomized_class = randomize_matrix(this_class)
            by_class[i] = randomized_class
        return by_class


def randomize_matrix(matrix):
    randomized = []
    i = 0
    while len(matrix) > 0:  # move random rows until empty
        j = len(matrix)
        selected = np.random.randint(i, j)
        row = matrix.pop(selected)
        randomized.append(row)
    return randomized


def z_standardize(sd, mean, x):
    """formula for z-standardization"""
    z = (float(x) - float(mean)) / float(sd)
    return z


class CrossValidator:
    """class to cross-validate data into k folds"""

    def __init__(self, data, k, prune=False):
        if data.operation == 1:  # for regression data
            self.ordered = True
        else:
            self.ordered = False
        self.k = k
        self.data = data  # appends the data object
        self.folds = []
        self.results = []  # holds the results / prediction
        self.folds_standards = []
        self.training_sets = None
        self.test_sets = None
        self.prune_set = None
        if prune:
            self.fold_size = data.records // k + 1
        else:
            self.fold_size = data.records // k
        self.standards = None
        self.prior = None
        self.create_folds()

    def create_folds(self):
        for i in range(self.k):
            self.folds.append([])
        by_class = self.data.split_by_class_binary()
        i = 0
        for this_class in by_class:
            while len(this_class) > 0:
                rand = np.random.randint(0, len(this_class))
                record = this_class.pop(rand)
                if i == self.k:
                    i = 0
                self.folds[i].append(record)
                i += 1
        i = 0
        self.training_sets = []
        self.test_sets = []
        while i < self.k:
            training_set = []
            test_set = []
            for f, fold in enumerate(self.folds):
                if f is not i:
                    training_set = training_set + fold
                if f is i:
                    test_set = fold
            self.training_sets.append(training_set)
            self.test_sets.append(test_set)
            i += 1
        return self.training_sets, self.test_sets

    def generate_random_array(self):
        """function to generate random arrays for more random ordered stratification"""
        numbers = []
        random_numbers = []
        i = 0
        while i < self.k:  # create 0 to n array
            numbers.append(i)
            i += 1
        while len(numbers) > 0:  # randomize 0 to n array
            j = len(numbers)
            random_choice = np.random.randint(0, j)
            random_numbers.append(numbers.pop(random_choice))
        return random_numbers


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
