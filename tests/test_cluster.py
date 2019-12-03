import unittest

from chess.manifold import *


class TestCluster(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = np.random.randn(100, 100)
        cls.manifold = Manifold(cls.data, 'euclidean')
        cls.allsame = Manifold(np.ones((100, 100)), 'euclidean')
        cls.cluster = Cluster(cls.manifold, cls.manifold.argpoints, '')
        return

    def test_init(self):
        Cluster(self.manifold, self.manifold.argpoints, '')
        with self.assertRaises(ValueError):
            Cluster(self.manifold, [], '')
        return

    def test_points(self):
        self.assertTrue(np.array_equal(
            self.manifold.data,
            self.cluster.points
        ))
        return

    def test_argpoints(self):
        self.assertListEqual(
            self.manifold.argpoints,
            self.cluster.argpoints
        )
        return

    def test_samples(self):
        pass

    def test_argsamples(self):
        pass

    def test_nsamples(self):
        self.assertEqual(
            np.sqrt(len(self.data)),
            self.cluster.nsamples
        )
