import points as pnt


def merge_sort(points):  # function for merge sort on an instance
    length = len(points)
    if length == 1:  # base case for recursion
        return points
    else:  # divide the plane into two
        mid = int(length / 2)  # find the mid-point
        l_list = merge_sort(points[:mid])  # create a list for left half
        r_list = merge_sort(points[mid:])  # create a list for right half
        return merge(l_list, r_list)  # merges the sorted list


def merge(l, r):  # function for merging sorted lists
    i = 0  # initialize counters
    j = 0
    k = 0
    merged = []
    while i < len(l) and j < len(r):  # while statements to append the least value by X to plane
        try:
            if l[i].x < r[j].x:
                merged.append(l[i])  # add point from the left list
                i = i + 1
            else:
                merged.append(r[j])  # add point from the right list
                j = j + 1
            k = k + 1
        except AttributeError:
            l[i] = pnt.Point(l[i][0], l[i][1])
            r[j] = pnt.Point(r[j][0], r[j][1])
    while i < len(l):  # executes if there are left over points on the left
        merged.append(l[i])
        i = i + 1
        k = k + 1
    while j < len(r):  # executes if there are left over points on the right
        merged.append(r[j])
        j = j + 1
        k = k + 1
    # print("finished sorting in %s cycles." % k)  # calculates how long it takes to execute each merge
    return merged
