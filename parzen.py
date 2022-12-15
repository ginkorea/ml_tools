import numpy as np
import matplotlib.pyplot as plt


class Kernel:

    def __init__(self, location):
        self.location = location

    def at_x(self, x):
        y = (1 / (np.sqrt(2 * np.pi))) * np.e ** -(((x - self.location) ** 2) / 2)
        return y


class PW:

    def __init__(self, points):
        self.n = len(points)
        self.kernels = []
        for point in points:
            this_kernel = Kernel(point)
            self.kernels.append(this_kernel)

    def density_at(self, x):
        density = 0
        for kernel in self.kernels:
            density += kernel.at_x(x)
            print(kernel.at_x(x))
        density = density / self.n
        return density

    def plot(self, start=0, end=20, x_inc=0.1):
        x_array = []
        y_array = []
        x = start
        while x < end:
            x_array.append(x)
            y_array.append(self.density_at(x))
            x += x_inc
        plt.title("Parzen Window")
        plt.plot(x_array, y_array)
        plt.show()


data = [10.94, 11.75, 11.48, 7.681, 11.98, 12.75, 7.556, 11.20, 13.03, 11.23]

window = PW(data)
window.plot()
at = 12
at_12 = window.density_at(at)
print("density at %s is: %s" % (at, at_12))
