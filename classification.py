import numpy as np
import graph
import points as pnt

"""module of tools for classification"""


def condensed_knn_classification(matrix):
    """performs condensed knn classification from a matrix of points in two dimensions"""
    z = []
    # w = pnt.transform_to_points(matrix, 0, 1, 2)
    w = matrix
    continue_processing = True
    while continue_processing:
        changed = False
        w = randomize_matrix(w)
        i = 0
        for point in w:
            try:
                point_index, closest_point, distance = point.calculate_closest_distance(z)
                # print("point index = %s closest_point = %s  distance = %s" % (point_index, closest_point, distance))
                if closest_point.classification != point.classification:
                    z.append(w.pop(i))
                    changed = True
                else:
                    i += 1
            except AttributeError:
                # z is empty
                z.append(w.pop(i))
                changed = True
        if changed:
            continue_processing = True
        else:
            continue_processing = False
    return z


def knn_voting_classification(training, test, k=5, class_1="E", class_2="F"):
    """performs knn voting classification from a training and test matrix of points """
    z = training
    w = test
    for point in w:  # iterates through points in w
        c1_count = 0
        c2_count = 0
        k_nearest = point.find_knn_voting(z, k)
        for nearest in k_nearest:
            if nearest.classification == class_1:
                c1_count += 1
            elif nearest.classification == class_2:
                c2_count += 1
        # print("c1 = %s c2 = %s" % (c1_count, c2_count))
        if c1_count > c2_count:  # determines which classification it should be
            point.prediction = class_1
        elif c2_count > c1_count:
            point.prediction = class_2
        else:
            rand = np.random.randint(1, 2)
            if rand == 1:
                point.prediction = class_1
            elif rand == 2:
                point.prediction = class_2
    return w


def randomize_matrix(matrix):
    """performs randomization, takes matrix creates a new matrix with rows in random order"""
    randomized = []
    i = 0
    copy = matrix  # create a copy
    while len(copy) > 0:  # move random rows until empty
        j = len(copy)
        pop = np.random.randint(i, j)
        row = copy.pop(pop)
        randomized.append(row)
    return randomized  # matrix is now randomized


def prior_probability(matrix, class1="E", class2="F"):
    """determines the prior probability from the matrix"""
    c1 = 0
    c2 = 0
    total = 0
    for p in matrix:
        # print(p.classification)
        if p.classification == class1:
            c1 += 1
            # print(class1)
        elif p.classification == class2:
            c2 += 1
            # print(class2)
        total += 1
    prior_c1 = c1 / total
    prior_c2 = c2 / total
    print("c1 total = %s c2 total = %s" % (c1, c2))
    prior = [prior_c1, prior_c2]
    return prior


def calculate_total_error(matrix, prior, class1="E", class2="F"):
    """calculates the total error from a classification.  takes a processed matrix of points as input"""
    c1 = 0
    c2 = 0
    e1 = 0
    e2 = 0
    for p in matrix:
        if p.classification == class1:
            c1 += 1
            if p.classification != p.prediction:
                e2 += 1
        elif p.classification == class2:
            c2 += 1
            if p.classification != p.prediction:
                e1 += 1
    ec1 = (e2 / c1) * prior[0]
    ec2 = (e1 / c2) * prior[1]
    total_error = ec1 + ec2
    return total_error


def c1_classification_prep(cv_data):
    """tools to prepare the data for cross-fold testing and training"""
    c1 = cv_data
    full_folds = []
    for fold in c1.folds:
        training = fold.training + fold.test
        full_folds.append(training)


def hyper_tune_knn_classification(training, test, prior, low_k=3, i_k=2, hi_k=45, class1="E", class2="F"):
    """iterates through a set to determine the total error based on different number of knn voting members"""
    knn = low_k
    total_error_array = []
    while knn < hi_k:
        matrix = knn_voting_classification(training, test, k=knn, class_1=class1, class_2=class2)
        total_error = calculate_total_error(matrix, prior, class1, class2)
        total_error_array.append(pnt.Point(knn, total_error))
        knn = knn + i_k
    return total_error_array


def project_1_classification(cv_data):
    """performs all required classification tasks for project 1 on dataset c1"""
    training, test = c1_classification_prep(cv_data)
    c1_points = pnt.transform_to_points(cv_data.data.matrix, 0, 1, ci=2)
    c1_test1 = pnt.transform_to_points(cv_data.data.h, 0, 1, ci=2)
    c1_test2 = pnt.transform_to_points(cv_data.data.h, 0, 1, ci=2)
    c1_test3 = pnt.transform_to_points(cv_data.data.h, 0, 1, ci=2)
    cv_data.prior = prior_probability(c1_points)
    hyper_tuned_k = project_1_quick_tune_find_total_error(c1_points, c1_test1, cv_data.prior, low_k=5, i_k=10, hi_k=36)
    # hyper_tuned_k = pnt.Point(25, 0.0386)
    knn_5 = knn_voting_classification(c1_points, c1_test2, k=5)
    hyper_tuned_knn = knn_voting_classification(c1_points, c1_test3, k=hyper_tuned_k.x)
    class1, class2 = pnt.array_by_class(c1_points)
    knn_5c1, knn_5c2 = pnt.array_by_class(knn_5, prediction=True)
    htk_c1, htk_c2 = pnt.array_by_class(hyper_tuned_knn, prediction=True)
    graph.graph_points_classification(class1, class2, cv_data)
    knn_5_error = calculate_total_error(knn_5, cv_data.prior)
    graph.graph_points_classification(knn_5c1, knn_5c2, cv_data, knn=5, error=knn_5_error)
    graph.graph_points_classification(htk_c1, htk_c2, cv_data, knn=hyper_tuned_k.x, error=hyper_tuned_k.y)
    for i, t in enumerate(training):
        results = knn_voting_classification(t, test[i], k=hyper_tuned_k.x)
        total_error = calculate_total_error(results, cv_data.prior)
        c1, c2 = pnt.array_by_class(results, prediction=True)
        graph.graph_points_classification(c1, c2, cv_data, knn=hyper_tuned_k.x, error=total_error)


def project_1_hypertune_find_total_error(training, test, k, prior, low_k=3, i_k=24, hi_k=99):
    """iterates through all training sets to find total error"""
    i = 0
    j = k
    total_error_matrix = []
    averages = []
    hyper_tuned_k = pnt.Point(np.infty, np.infty)
    while i < j:
        print("hyper tuning knn voting classification")
        total_error_array = hyper_tune_knn_classification(training[i], test[i], prior, low_k=low_k, i_k=i_k, hi_k=hi_k)
        total_error_matrix.append(total_error_array)
        i += 1
    i = 0
    while i < len(total_error_matrix[0]):
        total = 0
        for a in total_error_matrix:
            total = total + a[i].y
            knn = a[i].x
        average = total / len(total_error_matrix)
        averages.append(pnt.Point(knn, average))
        i += 1
    for e in averages:
        print("k= %s, mse= %s" % (e.x, e.y))
        if e.y < hyper_tuned_k.y:
            hyper_tuned_k = e
    print("best k value is: %s with a mse of %s" % (hyper_tuned_k.x, hyper_tuned_k.y))
    return hyper_tuned_k


def project_1_quick_tune_find_total_error(training, test, prior, low_k=3, i_k=24, hi_k=99, class1="E", class2="F"):
    """developed to use the hyper_tuning set, due to processing time
    performs hyper-tuning at various different values of k to determine the
    lowest total error value for knn voting"""
    hyper_tuned_k = pnt.Point(np.infty, np.infty)
    print("(quick) hyper tuning knn voting classification")
    print(class1)
    print(class2)
    total_error_array = hyper_tune_knn_classification(training, test, prior, low_k=low_k, i_k=i_k, hi_k=hi_k,
                                                      class1=class1, class2=class2)
    for e in total_error_array:
        print("k= %s, total error= %s" % (e.x, e.y))
        if e.y < hyper_tuned_k.y:
            hyper_tuned_k = e
    print("best k value is: %s with a total error of %s" % (hyper_tuned_k.x, hyper_tuned_k.y))
    return hyper_tuned_k


