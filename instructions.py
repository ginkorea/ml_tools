"""set of utilities and instructions for handling the data"""


def fill_missing_data(data):
    """fills in the missing data in a matrix"""
    c = 1
    data.means.append(None)
    while c < data.columns:
        value = 0
        e = 0
        replace = []
        for i, r in enumerate(data.matrix):
            try:
                value += int(r[c])
            except ValueError:
                e += 1
                replace.append(i)
        value = int(value / (data.records - e))
        data.means.append(value)
        if e > 0:
            for r in replace:
                data.matrix[r][c] = data.means[-1]
            print("replaced %s values" % len(replace))
        c += 1


def normalize_binary_classifier(data):
    """function to normalize binary data i.e. benign or malignant tumors"""
    print("normalizing binary data")
    classifier = int(data.instructions[1])
    false = data.instructions[2]
    true = data.instructions[3]
    for row in data.matrix:
        if row[classifier] == false:
            row[classifier] = 0
        elif row[classifier] == true:
            row[classifier] = 1


def normalize_nominal_classifier(data):
    """function to normalize nominal classifiers, hard coded for the voting data set"""
    print("normalizing nominal data")
    for m in data.matrix:  # iterates through matrix and replaces text values with nominal values
        for n, v in enumerate(m):
            if m[n] in ["democrat", "y"]:
                m[n] = [1, 0, 0]
            elif m[n] in ["republican", "n"]:
                m[n] = [0, 1, 0]
            elif m[n] in ["?"]:
                m[n] = [0, 0, 1]


def normalize_ordinal_classifier(data):
    """function to normalize ordinal data, hard coded for the car data set"""
    print("normalizing ordinal data")
    for m in data.matrix:  # iterates through matrix and replaces text values with ordinal data
        for n, v in enumerate(m):
            if m[n] in ["unacc", "low", "small"]:
                m[n] = 0
            elif m[n] in ["acc", "med"]:
                m[n] = 1
            elif m[n] in ["good", "high", "big"]:
                m[n] = 2
            elif m[n] in ["vgood", "vhigh"]:
                m[n] = 3
            elif m[n] in ["more", "5more"]:
                m[n] = 4


def discard_columns(data):
    """function for discarding columns at the beginning of a matrix"""
    print("discarding columns")
    start = int(data.instructions[2])  # hard coded location for column to start on
    for i, m in enumerate(data.matrix):
        data.matrix[i] = m[start:]  # take slice of matrix


def get_instructions(data):
    """function for getting the instructions for processing the dataset"""
    numb, classifier = int(data.instructions[0]), int(data.instructions[1])  # position 0 is handling instructions
    # position 1 is for the row of importance (i.e. classification or regression)
    return numb, classifier


def get_pie_labels(data):
    """function for generating labels for a pie chart, hard coded for specific datasets"""
    numb, classifier = get_instructions(data)
    labels = []
    if numb == 0:
        labels = ["benign", "malignant"]
    elif numb == 1:
        labels = ["unacceptable", "acceptable", "good", "very good"]
    elif numb == 2:
        labels = ["democrat", "republican", "independent"]
    length = len(labels)
    return length, labels
