import instructions as inst
import numpy as np
import regression as rg
import classification as cl


def get_column_array(matrix, column):
    """function to get a column from a matrix and return a numpy array"""
    array = []
    for row in matrix:  # iterates through matrix and creates an array of floats
        array.append(float(row[column]))
    array = np.array(array)
    return array


def convert_column_to_float(matrix, column):
    """function to convert data in a column into a float"""
    array = []
    for row in matrix:  # iterates through matrix and replaces it with float values
        try:
            row[column] = float(row[column])
            array.append(row)
        except TypeError:
            print("Type Error: unable to parse %s to float" % row[column])
    return array  # returns column


def z_standardize_data(training_standards, test_set, regress):
    """function to perform z-standardization on a column in a matrix"""
    sd = training_standards.sd  # standard deviation from training set
    mean = training_standards.mean  # standard deviation from test set
    for row in test_set:  # iterates through the test set and z standardizes it according to standards in training set.
        x = row[regress]
        try:
            row[regress] = z_standardize(sd, mean, x)
        except ZeroDivisionError:
            continue


def z_standardize(sd, mean, x):
    """formula for z-standardization"""
    z = (float(x) - float(mean)) / float(sd)
    return z


class CrossValidator:
    """class to cross-validate data into k folds"""

    def __init__(self, data, k, prune=True):
        if data.operation == 1:  # for regression data
            self.ordered = True
        else:
            self.ordered = False
        self.k = k
        self.data = data  # appends the data object
        self.stratify()  # performs ordered or random stratification
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
        self.z = None
        self.prior = None
        self.create_folds()

    def classify(self):
        """function to perform naive classification"""
        numb, classifier = inst.get_instructions(self.data)
        print("instruction %s classifier %s" % (numb, classifier))
        if numb in [0]:  # breast cancer data set
            print("nothing required here. code is functioning elsewhere")
        if numb in [1]:  # car dataset
            print("nothing required here. code is functioning elsewhere")
        if numb in [2]:  # voting data
            for fold in self.folds:
                results = [0, 0]  # [democrat, republican]
                for record in fold.training:
                    classification = record[classifier]
                    index = classification.index(1)
                    results[index] = results[index] + 1
                value = max(results)  # find the majority to predict
                index = results.index(value)
                self.results.append(index)
        if numb in [20]:
            cl.project_1_classification(self)

    def stratify(self):
        """function to perform random or ordered stratification"""
        if self.data.operation == 0:  # classification
            self.ordered_stratification()
        elif self.data.operation == 1:  # regression
            self.random_stratification()

    def random_stratification(self):
        """performs random stratification, takes matrix creates a new matrix with rows in random order"""
        randomized = []
        i = 0
        copy = self.data.matrix  # create a copy
        while len(copy) > 0:  # move random rows until empty
            j = len(copy)
            pop = np.random.randint(i, j)
            row = copy.pop(pop)
            randomized.append(row)
        self.data.matrix = randomized  # matrix is now randomized

    def ordered_stratification(self):
        """performs ordered stratification to ensure a generally equal distribution among k-folds and sets"""
        numb, column = inst.get_instructions(self.data)
        self.data.matrix = convert_column_to_float(self.data.matrix, column)
        ordered = []
        split = []
        for k in range(self.k):  # create k arrays
            split.append([])
        i = 0
        copy = sorted(self.data.matrix, key=lambda x: x[column])  # sort on column of significance
        random_array = generate_random_array(self.k)  # create a random array from 0 to k
        for row in copy:  # iterates through the copy and divides the matrix into k arrays
            if i < self.k:
                j = random_array[i]
                split[j].append(row)
                i += 1
            else:
                i = 0
                random_array = generate_random_array(self.k)
                j = random_array[i]
                split[j].append(row)
                i += 1
        for sp in split:  # recombines the matrix
            for row in sp:
                ordered.append(row)
        self.data.matrix = ordered
        print("length of new matrix: %s" % len(self.data.matrix))

    def regression(self):
        """function to perform regression on a dataset"""
        numb, regress = inst.get_instructions(self.data)
        if numb in [3, 4, 5, 21]:  # all regression problem sets
            column = get_column_array(self.data.matrix, regress)
            self.standards = Standards(column)  # calculates the mean and sd for entire dataset
            print("mean = %s" % self.standards.mean)
            print("standard deviation = %s" % self.standards.sd)
            for i, fold in enumerate(self.folds):
                column = get_column_array(fold.training, regress)  # grabs array of regression column in training set
                standard = Standards(column)  # calculates the mean and sd based on the training set
                self.folds_standards.append(standard)
                self.results.append(standard.mean)
                if numb in [5]:  # forest fire data set
                    print("z-standardizing data for fold %s" % str(i + 1))
                    z_standardize_data(standard, fold.test, regress)
            for i, standard in enumerate(self.folds_standards):  # displays mean and sd for each fold
                print("fold %s: mean = %s, sd = %s" % (i + 1, standard.mean, standard.sd))
        if numb in [21]:
            rg.project_1_regression(self)

    def create_folds(self):
        classes = []
        for i in range(self.k + 1):
            self.folds.append([])
        for i in range(4):  # of classes
            classes.append([])
        for record in self.data.matrix:
            class_index = int(record[int(self.data.instructions[1])])
            classes[class_index].append(record)
        i = 0
        for class_index in classes:
            while len(class_index) > 0:
                rand = np.random.randint(0, len(class_index))
                record = class_index.pop(rand)
                if i == self.k + 1:
                    i = 0
                self.folds[i].append(record)
                i += 1
        i = 0
        prune_set = self.folds.pop(0)
        j = len(self.folds)
        training_sets = []
        test_sets = []
        while i < j:
            training_set = []
            test_set = []
            for f, fold in enumerate(self.folds):
                if f is not i:
                    training_set = training_set + fold
                if f is i:
                    test_set = fold
            training_sets.append(training_set)
            test_sets.append(test_set)
            i += 1
        self.training_sets = training_sets
        self.test_sets = test_sets
        self.prune_set = prune_set
        return training_sets, test_sets, prune_set


class Standards:
    """class to hold mean and standard deviation for a dataset"""
    mean = None
    sd = None

    def __init__(self, array):
        self.mean = np.mean(array)
        self.sd = np.std(array)


class Set:
    """class to divide a fold into training, test, and optionally validation sets"""
    training = None
    test = None
    validation = None
    z = None

    def __init__(self, matrix, validation=False, ordered=False):
        if not ordered:
            i = 0
            if not validation:  # divide matrix into training and test set
                j = len(matrix) // 2
                self.test = matrix[j:]  # last half
            else:  # divide matrix into training, test, and validation sets
                j = len(matrix) // 3
                k = j * 2
                self.test = matrix[j:k]  # 1st third
                self.validation = matrix[k:]  # last third
            self.training = matrix[i:j]  # 1st section (half or third)
        else:
            self.training = []
            self.test = []
            if validation:
                self.validation = []
                self.n = 3
            else:
                self.n = 2
            i = 0
            random_array = generate_random_array(self.n)
            while len(matrix) > 0:  # appends data to test, training, and validation sets in random order
                row = matrix.pop(0)
                j = random_array[i]
                if j == 0:
                    self.training.append(row)
                elif j == 1:
                    self.test.append(row)
                elif j == 2:
                    self.validation.append(row)
                if i < (self.n - 1):
                    i += 1
                else:
                    i = 0
                    random_array = generate_random_array(self.n)


def generate_random_array(n):
    """function to generate random arrays for more random ordered stratification"""
    numbers = []
    random_numbers = []
    i = 0
    while i < n:  # create 0 to n array
        numbers.append(i)
        i += 1
    while len(numbers) > 0:  # randomize 0 to n array
        j = len(numbers)
        random_choice = np.random.randint(0, j)
        random_numbers.append(numbers.pop(random_choice))
    return random_numbers
