import numpy as np
from windrose import WindroseAxes


def exp_kernel(x, sigma=1.):
    # für die Glättung bei der match-to-observation
    return np.exp(-x ** 2 / (2 * sigma ** 2))


def custom_gaussian_filter(x, sigma=1., truncate=4.):
    radius = truncate
    kernel = exp_kernel(np.arange(-radius, radius + 1), sigma=sigma)
    # print(f"kernel: {kernel}")
    kernel = kernel / kernel.sum()
    # print(f"kernel/kernel.sum:{kernel}")
    ysmooth = np.convolve(kernel, x, mode="same")
    return ysmooth


def arctan(u, v):
    # def arctan(v,u):
    angle = np.arctan2(v, u)
    shift = (angle + 2 * np.pi) % (2 * np.pi)
    winkel = shift * 180.0 / np.pi
    return winkel


# Funktion windrosen definieren
def windrose(direction, speed, title):  # , savename):#, savename):#, savename):
    ax = WindroseAxes.from_ax()
    ax.bar(direction, speed, normed=True, opening=0.8, edgecolor="white")
    ax.set_title(title)
    ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE', ])
    ax.set_legend()


def filter_nans(data):
    return np.ma.array(data, mask=np.isnan(data))


def count_nans(data):
    print(np.isnan(data).astype(int).sum())


def linear(x, a, b):
    return a * x + b


def quad(x, a, b, c):
    return a * x ** 2 + b * x + c


def linear_error(x, cov):
    grad_y = np.array([x, 1])
    delta_y = np.sqrt(grad_y.dot((cov @ grad_y)))
    return delta_y
