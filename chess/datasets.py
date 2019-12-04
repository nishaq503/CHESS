from typing import Union, Tuple, List

import numpy as np

np.random.seed(42)

Data = Union[np.ndarray, np.memmap]
Label = Union[np.ndarray, List[int]]


def ring_data(num_points: int, radius: float, noise: float) -> np.ndarray:
    theta: np.ndarray = 2 * np.pi * np.random.rand(num_points)
    x: np.ndarray = radius * np.cos(theta) + noise * np.random.randn(num_points)
    y: np.ndarray = radius * np.sin(theta) + noise * np.random.randn(num_points)
    return np.stack([x, y], axis=1)


def bullseye(num_rings: int = 4, n: int = 1_000) -> Tuple[np.ndarray, List[int]]:
    data: np.ndarray = np.ndarray(shape=(0, 2))
    labels: List[int] = list()
    for r in range(1, 2 * num_rings, 2):
        ring: np.ndarray = ring_data(num_points=n * r, radius=r, noise=0.1)
        labels.extend([r for _ in range(n * r)])
        data = np.concatenate([data, ring], axis=0)
    return data, labels


def line(num_points: int = 10_000, m: float = 1, c: float = 0., noise: float = 0.05) -> Tuple[np.ndarray, List[int]]:
    x = np.random.rand(num_points)
    y = m * x + c
    data = np.asarray((x, y)).T
    data = data + np.random.rand(*data.shape) * noise
    labels = np.ones_like(x.T)
    return data, list(labels)


def xor(num_points: int = 10_000) -> Tuple[np.ndarray, List[int]]:
    data = np.random.rand(num_points, 2)
    labels = [int((x > 0.5) != (y > 0.5)) for x, y, in data]
    return data, labels


def spiral_2d(num_points: int = 10_000, noise: float = 0.25) -> Tuple[np.ndarray, List[int]]:
    theta = np.sqrt(np.random.rand(num_points)) * 2 * np.pi

    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(num_points, 2) * noise

    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(num_points, 2) * noise

    data = np.concatenate([x_a, x_b])
    labels = list(np.concatenate([np.zeros(len(x_a)), np.ones(len(x_a))]))
    return data, labels


def generate_torus(num_points: int, r_tube: float, r_torus: float, noise: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v = np.random.rand(num_points), np.random.rand(num_points)
    u, v = u * 2 * np.pi, v * 2 * np.pi
    x = (r_torus + r_tube * np.cos(v)) * np.cos(u) + (np.random.randn(num_points) * noise)
    y = (r_torus + r_tube * np.cos(v)) * np.sin(u) + (np.random.randn(num_points) * noise)
    z = r_tube * np.sin(v) + (np.random.randn(num_points) * noise)
    return x, y, z


def tori(num_points: int = 20_000, noise: float = 0.1) -> Tuple[np.ndarray, List[int]]:
    x, y, z = generate_torus(num_points=num_points // 2, r_tube=1., r_torus=5., noise=noise)
    torus_1 = np.stack([x - 5., y, z], axis=1)
    labels = [0 for _ in x]

    x, y, z = generate_torus(num_points=num_points // 2, r_tube=1., r_torus=5., noise=noise)
    torus_2 = np.stack([x, z, y], axis=1)
    labels.extend([1 for _ in x])

    data = np.concatenate([torus_1, torus_2], axis=0)
    return data, labels


def spiral_3d(num_points: int, radius: float, height: float, num_turns: int, noise: float) -> np.ndarray:
    theta: np.ndarray = 2 * np.pi * np.random.rand(num_points) * num_turns
    x: np.ndarray = radius * np.cos(theta) + noise * np.random.randn(num_points)
    y: np.ndarray = radius * np.sin(theta) + noise * np.random.randn(num_points)
    z: np.ndarray = height * theta / (2 * np.pi * num_turns) + noise * np.random.randn(num_points)
    return np.stack([x, y, z], axis=1)


def line_3d(num_points: int, height: float, noise: float) -> np.ndarray:
    x: np.ndarray = noise * np.random.randn(num_points)
    y: np.ndarray = noise * np.random.randn(num_points)
    z: np.ndarray = height * np.random.rand(num_points)
    return np.stack([x, y, z], axis=1)


def skewer(num_points: int = 12_000, radius: float = 1., height: float = 5., num_turns: int = 5, noise: float = 0.05) -> Tuple[np.ndarray, List[int]]:
    spiral_points, line_points = 5 * num_points // 6, num_points // 6
    spiral_data = spiral_3d(spiral_points, radius, height, num_turns, noise)
    labels = [0 for _ in range(spiral_data.shape[0])]

    line_data = line_3d(line_points, height, noise)
    labels.extend([1 for _ in range(line_data.shape[0])])

    data = np.concatenate([spiral_data, line_data], axis=0)
    return data, labels


def plot():
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    data, labels = skewer()
    print(data.shape, len(labels))
    # limits = [int(np.min(data) - 1), int(np.max(data) + 1)]

    for azimuth in range(0, 180, 6):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=0.1)
        plt.axis('off')
        # ax.set_xlim(limits)
        # ax.set_ylim(limits)
        # ax.set_zlim(limits)
        ax.view_init(elev=azimuth, azim=azimuth)
        plt.show()
    return


if __name__ == '__main__':
    plot()
