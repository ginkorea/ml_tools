import numpy as np


def calc_y(w, x):
    a = np.dot(w.transpose(), x)
    return a


def sgn(y):
    if y > 0:
        a = 1
    else:
        a = -1
    return a


def incremental_grad_descent_step(t, y, array_x, array_w, n):
    for i, w in enumerate(array_w):
        array_w[i] = array_w[i] + (n * (t - y) * array_x[i])
    return array_w


def walk_incremental_grad(array_x, w, t, n):
    for i, x in enumerate(array_x):
        y = calc_y(w, x)
        print("y: %s" % y)
        sg = sgn(y)
        print("sgn: %s" % sg)
        w = incremental_grad_descent_step(t[i], y, x, w, n)
        print("w%s: %s" % (i + 1, w))


def grad_descent(array_x, w, t, n):
    t_minus_y = []
    for i, x in enumerate(array_x):
        y = calc_y(w, x)
        print(y)
        t_minus_y.append((t[i] - y))
    for i, wi in enumerate(w):
        sums = 0
        for m, t, in enumerate(t_minus_y):
            sums = sums + (t_minus_y[m] * array_x[m][i])
        w[i] = wi + (n * sums)
    print(w)
    return w


def update_weight(weight, n, error, x):
    weight = weight + (n * error * x)
    return weight


def calculate_output_error_sigmoid(z, t):
    error = z * (1 - z) * (t - z)
    return error


def calculate_backprop(error, z, weight):
    backprop = error * weight * (z * (1 - z))
    return backprop


def sigmoid(y):
    sigmoid_y = 1 / (1 + np.e ** -y)
    return sigmoid_y


def set_state(previous_state, error, momentum, z):
    state = ((1 - momentum) * error * z) + (momentum * previous_state)
    return state


def add_momentum(n, state):
    weight_change = -n * state
    return weight_change


def update_weight_with_momentum(n, state, weight):
    new_weight = weight + add_momentum(n, state)
    return new_weight

def problem_52(wc, wd, point, target, state, m, n):
    cy = p.calc_y(wc, point)
    print("cy: %s" % cy)
    cz = p.sigmoid(cy)
    print("cz: %s" % cz)
    xc = [1, cz]
    dy = p.calc_y(wd, xc)
    print("dy: %s" % dy)
    dz = p.sigmoid(dy)
    print("dz: %s" % dz)
    de = p.calculate_output_error_sigmoid(dz, target)
    print("de at output: %s" % de)
    ce = p.calculate_backprop(de, cz, 0.1)
    print("ce hidden: %s" % ce)
    state[0] = p.set_state(state[0], ce, m, point[0])
    state[1] = p.set_state(state[1], ce, m, point[1])
    state[2] = p.set_state(state[2], ce, m, point[2])
    state[3] = p.set_state(state[3], de, m, xc[0])
    state[4] = p.set_state(state[4], de, m, xc[1])
    for i, w in enumerate(wc):
        wc[i] = p.update_weight_with_momentum(n, state[i], w)
    for i, w in enumerate(wd):
        wd[i] = p.update_weight_with_momentum(n, state[i + 3], w)
    print("weights \n c0: %f, ca: %f cb: %f" % (wc[0], wc[1], wc[2]))
    print("weights \n d0: %f, dc: %f" % (wd[0], wd[1]))
    return wc, wd, states
