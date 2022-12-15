import matplotlib.pyplot as plt
import instructions as inst
import manipulation as mn
import points as pnt
import MergeSort as ms

"""set of utilities to produce graphical results"""


def create_results_array(cv):
    """function to create the array to plot, returns x: fold label, y: prediction"""
    y = []
    x = []
    for i, result in enumerate(cv.results):
        column_name = "Fold " + str(i + 1)
        x.append(column_name)  # fold number
        y.append(float(result))  # prediction
    return x, y


def creat_results_pie(cv):
    """function to create a pie chart for classification problems"""
    k, labels = inst.get_pie_labels(cv.data)
    pie_data = []
    i = 0
    while i < k:  # create an array length of possible classifications
        pie_data.append(0)
        i += 1
    for result in cv.results:  # counts the number of times each item was predicted
        pie_data[result] += 1
    return pie_data, labels


def display_results(cv, option=0):
    """function that plots the results of a single dataset as a bar, line, or pie chart"""
    x, y = create_results_array(cv)
    if option == 0:
        plt.bar(x, y)
    elif option == 1:
        plt.plot(x, y, marker="o")
    elif option == 2:
        pie_data, labels = creat_results_pie(cv)
        plt.pie(pie_data, labels=labels)
        plt.legend()
    if option in [0, 1]:
        plt.ylabel("Prediction")
        plt.xlabel("Training Set")
    plt.title(cv.data.title)
    plt.show()


def show_results(database):
    """function that iterates through a list of datasets and produced graphs for all"""
    for dataset in database:
        display_results(dataset, option=0)  # bar chart
        display_results(dataset, option=1)  # line graph
        i, c = inst.get_instructions(dataset.data)
        if i in [0, 1, 2]:  # classification problems
            display_results(dataset, option=2)  # pie chart


def graph_2d(cv, xi=0, yi=1, ci=2):
    plt.title(cv.data.title)
    plt.ylabel(cv.data.names[yi])
    plt.xlabel(cv.data.names[xi])
    if cv.data.hyper_tuned:
        matrix = cv.data.h
    else:
        matrix = cv.data.matrix
    if cv.data.operation == 1:
        x = mn.get_column_array(matrix, xi)
        y = mn.get_column_array(matrix, yi)
        plt.scatter(x, y, marker=".", linewidths=.001)
    if cv.data.operation == 0:
        e = []
        f = []
        for m in matrix:
            if m[ci] == "E":
                e.append(m)
            elif m[ci] == "F":
                f.append(m)
        ex = mn.get_column_array(e, xi)
        ey = mn.get_column_array(e, yi)
        fx = mn.get_column_array(f, xi)
        fy = mn.get_column_array(f, yi)
        plt.scatter(ex, ey, c="red", marker=".", linewidths=.001, label="Class E")
        plt.scatter(fx, fy, c="blue", marker=".", linewidths=.001, label="Class F")
    plt.legend()
    plt.show()


def graph_points_classification(class1, class2, cv, xi=0, yi=1, knn=0, error=0, lab1="Class E", lab2="Class F"):
    if knn != 0:
        title = cv.data.title + " knn = " + str(knn)
        title = title + " error = " + str(error)
        plt.title(title)
    else:
        plt.title(cv.data.title)
    if lab1 == "Class E":
        plt.ylabel(cv.data.names[yi])
        plt.xlabel(cv.data.names[xi])
    else:
        plt.ylabel("Dimension 2")
        plt.xlabel("Dimension 1")
    x1, y1 = pnt.column_array_points(class1)
    x2, y2 = pnt.column_array_points(class2)
    plt.scatter(x1, y1, c="red", marker=".", linewidths=.001, label=lab1)
    plt.scatter(x2, y2, c="blue", marker=".", linewidths=.001, label=lab2)
    plt.legend()
    plt.show()


def graph_points_regression(training, test, predicted, cv=None, xi=0, yi=1, knn=None):
    training = ms.merge_sort(training)
    test = ms.merge_sort(test)
    predicted = ms.merge_sort(predicted)
    x1, y1 = pnt.column_array_points(training)
    x2, y2 = pnt.column_array_points(test)
    x3, y3 = pnt.column_array_points(predicted)
    if cv is not None:
        title = cv.data.title + " knn = " + str(knn)
        plt.title(title)
        plt.ylabel(cv.data.names[yi])
        plt.xlabel(cv.data.names[xi])
    plt.scatter(x1, y1, c="red", marker=".", label="training")
    plt.scatter(x2, y2, c="blue", marker=".", label="test")
    plt.plot(x3, y3, c="green", marker=".", label="predicted")
    plt.legend()
    plt.show()



