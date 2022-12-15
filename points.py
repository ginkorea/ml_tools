import math as m
import numpy as np


class Point:

    def __init__(self, x, y, z=None, classification=None):
        self.x = x
        self.y = y
        self.z = z
        self.classification = classification
        self.prediction = None
        self.closest = []
        self.closest_distance = [np.infty]

    def calculate_closest_distance(self, matrix, d=2):
        j = None
        k = len(self.closest)
        closest = None
        closest_distance = np.infty
        for i, point in enumerate(matrix):
            if d == 2:
                distance = calc_dist(self, point)
            elif d == 1:
                distance = calc_dist_1d(self, point)
            if 0 < distance < closest_distance:
                closest_distance = distance
                closest = point
                j = i
        self.closest.append(closest)
        self.closest_distance[k] = closest_distance
        self.closest_distance.append(np.infty)
        return j, closest, closest_distance

    def find_knn_voting(self, matrix, k, d=2):
        k_nearest = []
        copy = matrix.copy()
        while len(k_nearest) < k:
            j, closest, closest_distance = self.calculate_closest_distance(copy, d)
            nearest = copy.pop(j)
            k_nearest.append(nearest)
        return k_nearest


def calc_dist(p1, p2):  # calculates the distance between two points
    distance = m.sqrt((float(p2.x) - float(p1.x)) ** 2 + (float(p2.y) - float(p1.y)) ** 2)
    return distance


def calc_dist_1d(p1, p2):  # calculates the distance between two one dimensional points
    distance = m.sqrt((float(p2.x) - float(p1.x)) ** 2)
    return distance


def transform_to_points(matrix, xi, yi, ci=None):
    points = []
    for mx in matrix:
        if ci is not None:
            point = Point(float(mx[xi]), float(mx[yi]), classification=mx[ci])
        else:
            point = Point(float(mx[xi]), float(mx[yi]))
        points.append(point)
    return points


def array_by_class(matrix, class1="E", class2="F", prediction=False):
    c1 = []
    c2 = []
    for p in matrix:
        if prediction:
            classify = p.prediction
        else:
            classify = p.classification
        if classify == class1:
            c1.append(p)
        elif classify == class2:
            c2.append(p)
    return c1, c2


def column_array_points(points):
    x = []
    y = []
    for point in points:
        x.append(point.x)
        y.append(point.y)
    x = np.array(x)
    y = np.array(y)
    return x, y


def unravel_points_matrix(points):
    unraveled = []
    for point in points:
        row = [point.x, point.y, point.classification]
        unraveled.append(row)
    return unraveled


def reduce_dimensions_to_points(matrix, d1_start, d1_end, d2_start, d2_end, ci):
    points = []
    for mx in matrix:
        d1 = 0
        d2 = 0
        i = d1_start
        j = d1_end
        while i <= j:
            d1 = d1 + int(mx[i])
            i += 1
        i = d2_start
        j = d2_end
        while i <= j:
            d2 = d2 + int(mx[i])
            i += 1
        point = Point(float(d1), float(d2), classification=mx[ci])
        points.append(point)
    return points
