import numpy as np

"""functions to determine information gain"""

classes = [0, 1]  # classes in the cars dataset


def ensure_int_values(matrix):
    """used to ensure the values of a matrix are an integer"""
    for i, m in enumerate(matrix):
        for j, n in enumerate(m):
            matrix[i][j] = int(n)


def entropy(column):
    """takes a column as input, returns total entropy (ent) and per class entropy """
    class_totals, total = get_totals(column, 3)
    print(class_totals, total)
    ent, class_entropy = calculate_multi_class_entropy(class_totals, total)
    return ent, class_entropy


def calculate_multi_class_entropy(class_totals, total):
    """returns total entropy, entropy by class"""
    class_entropy = []
    class_probabilities = []
    for c in class_totals:
        class_probabilities.append(c / total)
    for p in class_probabilities:
        if p == 0:
            p = 1
        ent = - (p * np.log2(p))
        class_entropy.append(ent)
    ent = sum(class_entropy)
    return ent, class_entropy


def select_attribute(matrix, column, attribute):
    """returns selected, remainder, column, attribute"""
    selected = []
    copy = matrix.copy()
    for i, row in enumerate(copy):
        if int(row[column]) == attribute:
            selected.append(copy.pop(i))
    remainder = copy
    return selected, remainder, column, attribute


def get_totals(selected, ci, class_types=classes):
    """used to get the totals per class from a selected column"""
    class_totals = classes.copy()
    total = 0
    for i, c in enumerate(class_totals):
        class_totals[i] = 0
    for row in selected:
        try:
            row_class = row[ci]
        except TypeError:
            row_class = int(row)
        class_totals[row_class] += 1
        total += 1
    return class_totals, total


def count_values(column):
    """counts values in a column returns unique values, number of unique"""
    unique = []
    for m in column:
        if m not in unique:
            unique.append(m)
    number_of_unique = len(unique)
    unique.sort()
    return unique, number_of_unique


def get_column(matrix, column):
    """returns the column from a matrix"""
    return_column = []
    for m in matrix:
        return_column.append(int(m[column]))
    return return_column


def find_max_entropy(matrix, ci=6, class_types=classes):
    """takes a matrix as input.  optional class index and class types
    returns the column with the max entropy """
    n = ci - 1  # records end here
    max_entropy = -np.infty
    selected = None
    i = 0
    for m in range(n):  # iterates through columns
        column = get_column(matrix, i)
        unique_values, nu = count_values(column)  # gets unique attributes in column
        for attribute in unique_values:  # iteratively calculates entropy
            s, r, c, a = select_attribute(matrix, i, attribute)
            class_totals, totals = get_totals(s, ci, classes)
            ent = calculate_multi_class_entropy(class_totals, totals)
            if ent > max_entropy:
                max_entropy = ent  # this is the new max entropy
                selected = [max_entropy, s, r, c, a]  # s=selected, r=remainder, c=column, a=attribute
    return selected


def info_gain(s, sv, ent_s, ent_sv):
    """calculates the info gain given inputs of:
    \n s: number in set
    \n sv: number in selected attribute
    \n ent_s: entropy of s
    \n ent_sv: entropy of sv"""
    gain = ent_s - ((sv / s) * ent_sv)
    return gain


def find_gain(matrix, selected_attribute=0, column=0, class_index=6):
    """finds the gain of an attribute in a column"""
    class_column = get_column(matrix, class_index)
    ent_s, class_ent = entropy(class_column)
    s_total = len(matrix)
    s, r, c, a = select_attribute(matrix, column, selected_attribute)
    sv_total = len(s)
    sv_class_column = get_column(s, class_index)
    ent_sv, class_ent_sv = entropy(sv_class_column)
    print(s_total, sv_total, ent_s, ent_sv)
    gain = info_gain(s_total, sv_total, ent_s, ent_sv)
    ratio = sv_total / s_total
    return gain, ratio


def sum_column_gain(matrix, column, class_index):
    """sums the total of the column gain.  Takes a matrix, columm, and class_index and
    returns the total column gain"""
    gains = []
    selected_column = get_column(matrix, column)
    unique, numb_unique = count_values(selected_column)
    for attribute in unique:
        gain, ratio = find_gain(matrix, selected_attribute=attribute, column=column, class_index=class_index)
        weighted_gain = ratio * gain
        gains.append(weighted_gain)
    column_gain = sum(gains)
    return column_gain


def find_max_gain(matrix, class_index):
    """takes a matrix and class index as input, returns the column, gain, and gain per column"""
    number_columns = class_index - 1
    index = 0
    column_gains = []
    max_gain = -np.infty
    max_column = None
    while index < number_columns:
        column_gain = sum_column_gain(matrix, index, class_index)
        column_gains.append(column_gain)
        if column_gain > max_gain:
            max_gain = column_gain
            max_column = index
        index += 1
    return max_column, max_gain, column_gains
