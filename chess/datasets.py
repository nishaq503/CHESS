from typing import Union, Tuple, List

import numpy as np

np.random.seed(42)

Data = Union[np.ndarray, np.memmap]
Label = Union[np.ndarray, List[int]]


def ring(n: int = 10_000, scale: int = 12) -> Tuple[Data, Label]:
    samples: np.ndarray = scale * (np.random.rand(2, n) - 0.5)
    distances = np.linalg.norm(samples, axis=0)
    x = [samples[0, i] for i in range(n) if distances[i] < 2 or (4 < distances[i] < 6)]
    y = [samples[1, i] for i in range(n) if distances[i] < 2 or (4 < distances[i] < 6)]
    data: np.ndarray = np.asarray((x, y)).T
    labels = [0 if d < 2 else 1 for d in distances if d < 2 or (4 < d < 6)]
    return data, labels


def line(n: int = 10_000, scale: int = 12, shift=0.0) -> Tuple[Data, Label]:
    x = np.random.randn(1, n)
    y = scale * x + shift
    data = np.asarray((x, y)).T
    labels = np.ones(n, 1)
    return data, labels


def xor(n: int = 10_000) -> Tuple[Data, Label]:
    data = np.random.randn(n, 2)
    labels = [x > 0 != y > 0 for x, y, in data]
    return data, labels


def spiral(n: int = 10_000, noise: float = 1.0) -> Tuple[Data, Label]:
    theta = np.sqrt(np.random.rand(n)) * 2 * np.pi

    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(n, 2) * noise

    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(n, 2) * noise

    data = np.concatenate([x_a, x_b])
    labels = np.concatenate([np.zeros(n, 1), np.ones(n, 1)])
    return data, labels
