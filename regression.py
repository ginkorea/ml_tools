import numpy as np
import graph
import points as pnt

"""module to handle regression tasks in project 2"""


def gaussian_kernel(x, xt, h):
    """gaussian kernel function"""
    x = x.x
    kernel = (1 / np.sqrt(2 * np.pi)) * np.e ** ((-1 / 2 * ((x - xt) / h)) ** 2)
    return kernel


def knn_smoothing_function(p, knn, h):
    """performs knn smoothing of a gaussian kernel"""
    top = 0
    bottom = 0
    for k in knn:
        g = gaussian_kernel(p, k.x, h)
        top = top + (g * k.y)
        bottom = bottom + g
    predicted_y = top / bottom
    return predicted_y


def knn_gaussian_regression(training, test, k):
    """performs knn gaussian regression"""
    predicted = []
    for point in test:
        knn = point.find_knn_voting(training, k, d=1)
        h = find_h(point, knn)
        predicted_y = knn_smoothing_function(point, knn, h)
        predicted_point = pnt.Point(point.x, predicted_y)
        predicted.append(predicted_point)
    return predicted


def find_h(point, knn):
    """finds the distance h"""
    h = 0
    for p in knn:
        dist = pnt.calc_dist_1d(point, p)
        if dist > h:
            h = dist
    return h


def r1_regression_prep(cv_data):
    """preps data for regression tasks"""
    r1 = cv_data
    full_folds = []
    for fold in r1.folds:
        training = fold.training + fold.test
        training = pnt.transform_to_points(training, 0, 1)
        full_folds.append(training)
    i = 0
    j = len(cv_data.folds)
    training_sets = []
    test_sets = []
    while i < j:
        training_set = []
        test_set = []
        for f, fold in enumerate(full_folds):
            if f is not i:
                training_set = training_set + fold
            if f is i:
                test_set = fold
        training_sets.append(training_set)
        test_sets.append(test_set)
        i += 1
    return training_sets, test_sets


def average_regression(training_sets):
    """averages the regression value over multiple training sets for a smoother curve"""
    sets = len(training_sets)
    averages = []
    for i, p in enumerate(training_sets[0]):
        y = 0
        for dataset in training_sets:
            y = y + dataset[i].y
        y = y / sets
        x = p.x
        averages.append(pnt.Point(x, y))
    return averages


def square_error(point, predicted):
    """finds the square error"""
    sq_error = (point.y - predicted.y) ** 2
    return sq_error


def calculate_mean_square_error(test, training, k):
    """calculates the mean square error"""
    predicted = knn_gaussian_regression(training, test, k)
    n = len(predicted)
    sq_e = 0
    for i, predicted_point in enumerate(predicted):
        sq_e = sq_e + square_error(test[i], predicted_point)
    sq_e = sq_e / n
    return sq_e


def hyper_tune_knn_regression(test, training, low_k=3, iterate_k=2, hi_k=45):
    """performs knn hyper tuning over a test and training set"""
    k = low_k
    mse_array = []
    while k < hi_k:
        mse = calculate_mean_square_error(test, training, k)
        mse_array.append(pnt.Point(k, mse))
        k = k + iterate_k
    return mse_array


def project_1_regression(cv_data):
    """performs regression required for the project"""
    training, test = r1_regression_prep(cv_data)
    k = project_1_hypertune_find_mse(training, test, cv_data.k, low_k=9, iterate_k=20, hi_k=100)
    project_1_draw_regression_line(cv_data, training, knn=k.x)
    project_1_draw_regression_line(cv_data, training, knn=5)
    project_1_draw_regression_line(cv_data, training, knn=k.x//2)


def project_1_hypertune_find_mse(training, test, k, low_k=3, iterate_k=6, hi_k=99):
    """finds the mse for hyper tuning"""
    i = 0
    j = k
    hyper_tuned_k = pnt.Point(np.infty, np.infty)
    while i < j:
        print("hyper_tuning")
        mse = hyper_tune_knn_regression(test[i], training[i], low_k=low_k, iterate_k=iterate_k, hi_k=hi_k)
        i += 1
        for e in mse:
            print("k= %s, mse= %s" % (e.x, e.y))
            if e.y < hyper_tuned_k.y:
                hyper_tuned_k = e
    print("best k value is: %s with a mse of %s" % (hyper_tuned_k.x, hyper_tuned_k.y))
    return hyper_tuned_k


def project_1_draw_regression_line(cv_data, training, t_min=-1, t_iterate=.32, t_max=1, knn=21):
    i = 0
    j = cv_data.k
    t = t_min
    simple = []
    while t < t_max:
        point = pnt.Point(t, 0)
        simple.append(point)
        t = t + t_iterate
    predicted = []
    testing = pnt.transform_to_points(cv_data.data.h, 0, 1)
    while i < j:
        predicted.append(knn_gaussian_regression(training[i], testing, k=knn))
        i += 1
    averaged_prediction = average_regression(predicted)
    training_data = pnt.transform_to_points(cv_data.data.matrix, 0, 1)
    graph.graph_points_regression(training_data, testing, averaged_prediction, cv_data, knn=knn)
