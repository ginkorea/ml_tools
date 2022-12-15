# machine learning = 0, intro to ai = 1
# in person = 0, online = 1
# smith = 0, jones = 1
# no = 0, yes = 1

import math

import numpy as np

s1 = [0, 0, 0, 1]
s2 = [1, 1, 0, 1]
s3 = [1, 1, 1, 1]
s4 = [0, 1, 1, 0]
s5 = [0, 0, 0, 1]
s6 = [0, 1, 0, 1]
s7 = [1, 0, 0, 1]
s8 = [0, 0, 1, 0]
s9 = [1, 1, 0, 1]
s10 = [1, 0, 1, 1]
s11 = [0, 1, 0, 0]
sample = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11]


def prep_for_entropy(matrix, class_index=3):
    take = 0
    dont = 0
    total_records = len(matrix)
    for row in matrix:
        if row[class_index] == 0:
            dont += 1
        else:
            take += 1
    return total_records, dont, take


def entropy_s(neg, pos, total):
    try:
        p_pos = pos / total
    except ZeroDivisionError:
        p_pos = 1
    try:
        p_neg = neg / total
    except ZeroDivisionError:
        p_neg = 0
    if p_pos == 0:
        p_pos = 1
    elif p_neg == 0:
        p_neg = 1
    answer = -p_pos * math.log2(p_pos) - p_neg * math.log2(p_neg)
    return answer


def count(matrix, column, attribute, class_index=3):
    dont = 0
    take = 0
    for row in matrix:
        if row[column] == attribute:
            if row[class_index] == 0:
                dont += 1
            else:
                take += 1
    total_set = take + dont
    return total_set, dont, take


def calculate_entropy_gain_across(matrix, class_index=3):
    attributes = [0, 1]
    gains = []
    total_records, t_dont, t_take = prep_for_entropy(matrix)
    ent_s = entropy_s(t_dont, t_take, total_records)
    i = 0
    while i < 3:
        column_gain = 0
        for attribute in attributes:
            attribute_set, dont, take = count(matrix, i, attribute)
            this_entropy = entropy_s(dont, take, attribute_set)
            gain = info_gain(total_records, attribute_set, ent_s, this_entropy)
            column_gain += gain
            print("column %s, attribute %s, total %s, dont %s, take %s, entropy %s" %
                  (i, attribute, attribute_set, dont, take, this_entropy))
        print("column gain for column %s is %s" % (i, column_gain))
        gains.append(column_gain)
        i += 1
    return gains


def prune_set(matrix, column, attribute):
    to_pop = []
    for i, row in enumerate(matrix):
        if row[column] == attribute:
            to_pop.append(i)
    for value in reversed(to_pop):
        matrix.pop(value)
    return matrix


def info_gain(s, sv, ent_s, ent_sv):
    """calculates the info gain given inputs of:
    \n s: number in set
    \n sv: number in selected attribute
    \n ent_s: entropy of s
    \n ent_sv: entropy of sv"""
    gain = ent_s - ((sv / s) * ent_sv)
    return gain


def get_column(matrix, index):
    column = []
    for row in matrix:
        column.append(row[index])
    return column


def iterate(matrix):
    total_set, no, yes = prep_for_entropy(sample)
    print(total_set, no, yes)
    entropy = entropy_s(no, yes, total_set)
    print("entropy for set: %s" % entropy)
    gains = calculate_entropy_gain_across(sample)

"""
iterate(sample)
sample = prune_set(sample, 0, 1) # AI = yes
iterate(sample)
sample = prune_set(sample, 2, 1) # Jone = No
iterate(sample)
sample = prune_set(sample, 1, 0) # In Person = yes
iterate(sample)
print(sample)
"""

# 12 = correct
# 13 = correct
# 14 = correct
# 15 = correct
# 16 = incorrect
# 17 = correct
# 18 = correct
# 19 = incorrect