from time import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft
from scipy.stats import wasserstein_distance


def wave_form_distance(x: np.ndarray, y: np.ndarray):
    """
    Transforms x and y to the fourier domain and computes the
    Wasserstein distance between those curves.
    :param x: time-series data
    :param y: time-series data
    :return: distance between x and y
    """
    assert x.shape == y.shape, (x.shape, y.shape)
    assert len(x.shape) == 1, len(x.shape)

    xf, yf = np.abs(fft(x)[0:len(x) // 2]), np.abs(fft(y)[0:len(y) // 2])
    xf, yf = xf / np.linalg.norm(xf), yf / np.linalg.norm(yf)
    distance = np.float64(wasserstein_distance(xf, yf))
    return distance


def main():
    num_samples = list(range(1_000, 50_000, 1_000))
    sample_spacing = 1. / 1_000
    times = [0. for _ in num_samples]
    for i, n in enumerate(num_samples):
        x = np.linspace(0., n * sample_spacing, n)
        y1 = np.sin(50. * 2. * np.pi * x) + 0.5 * np.sin(80. * 2. * np.pi * x)
        y2 = np.sin(500. * 2. * np.pi * x) + 1.5 * np.sin(800. * 2. * np.pi * x)
        for _ in range(100):
            s = time()
            wave_form_distance(y1, y2)
            times[i] += (time() - s)
        times[i] /= 100
        print(f'n: {n}, t: {times[i]}')

    plt.plot(num_samples, times)
    plt.grid()
    plt.show()
    return


if __name__ == '__main__':
    main()
