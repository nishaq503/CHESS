import numpy as np


def ring():
    np.random.seed(42)
    scale, num_points = 12, 10_000
    samples: np.ndarray = scale * (np.random.rand(2, num_points) - 0.5)
    distances = np.linalg.norm(samples, axis=0)
    x = [samples[0, i] for i in range(num_points)
         if distances[i] < 6 and (distances[i] > 4 or distances[i] < 2)]
    y = [samples[1, i] for i in range(num_points)
         if distances[i] < 6 and (distances[i] > 4 or distances[i] < 2)]
    data: np.ndarray = np.asarray((x, y)).T
    labels = [0 if d < 2 else 1 for d in distances if d < 6 and (d > 4 or d < 2)]
    return data, labels
