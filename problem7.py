p1 = [-7.82, -4.58, -3.97]
p2 = [-6.68, 3.26, 2.71]
p3 = [4.36, -2.19, 2.09]
p4 = [6.72, 0.88, 2.80]
p5 = [-8.64, 3.06, 3.5]
p6 = [-6.87, 0.57, -5.45]
p7 = [4.47, -2.62, 5.76]
p8 = [6.73, -2.01, 4.18]
points_array = [p1, p2, p3, p4, p5, p6, p7, p8]

m1 = [1, 1, 1]
m2 = [-1, 1, -1]
means_array = [m1, m2]


def k_means(means, points):
    g1 = []
    g2 = []
    for point in points:
        closest = np.infty
        group = None
        for i, mean in enumerate(means):
            distance = math.dist(mean, point)
            if distance < closest:
                closest = distance
                group = i
        if group == 0:
            g1.append(point)
        elif group == 1:
            g2.append(point)
    return g1, g2


def return_mean_array(array, d=3):
    i = 0
    mean_array = []
    while i < d:
        this_sum = 0
        for point in array:
            this_sum += point[i]
        mean = this_sum / len(array)
        mean_array.append(mean)
        i += 1
    return mean_array


group_1, group_2 = k_means(means_array, points_array)

print(len(group_1))
print(group_1)
mean_1 = return_mean_array(group_1)
print(mean_1)
print(len(group_2))
print(group_2)
mean_2 = return_mean_array(group_2)
print(mean_2)