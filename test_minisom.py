import unittest

from numpy import all, array

# for unit test
from sklearn_minisom import SklearnMinisom


class TestMinisom(unittest.TestCase):
    def test_initialization(self):
        som = SklearnMinisom(
            x=2,
            y=10,
            sigma=2.0,
            learning_rate=0.7,
            num_iteration=100,
            neighborhood_function="test1",
            topology="test2",
            activation_distance="test3",
        )

        self.assertEqual(som.x, 2)
        self.assertEqual(som.y, 10)
        self.assertEqual(som.sigma, 2.0)
        self.assertEqual(som.learning_rate, 0.7)
        self.assertEqual(som.num_iteration, 100)
        self.assertEqual(som.neighborhood_function, "test1")
        self.assertEqual(som.topology, "test2")
        self.assertEqual(som.activation_distance, "test3")

    def test_fit(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = SklearnMinisom(x=3, y=3, sigma=1.0, learning_rate=0.5, num_iteration=1000)
        som.fit(X)

        self.assertIsNotNone(som.som)

    def test_transform(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = SklearnMinisom(x=3, y=3, sigma=1.0, learning_rate=0.5, num_iteration=1000)
        som.fit(X)
        transformed = som.transform(X)

        self.assertEqual(transformed.shape, (X.shape[0], 2))

    def test_predict(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = SklearnMinisom(x=3, y=3, sigma=1.0, learning_rate=0.5, num_iteration=1000)
        som.fit(X, y=[1, 2, 3, 4])
        predicted = som.predict(X)

        self.assertTrue(all(predicted == predicted.astype(int)))
        self.assertEqual(predicted.shape, (X.shape[0],))

    def test_fit_transform(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = SklearnMinisom(x=3, y=3, sigma=1.0, learning_rate=0.5, num_iteration=1000)
        transformed = som.fit_transform(X)

        self.assertEqual(transformed.shape, (X.shape[0], 2))

    def test_fit_transform_set_y(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = SklearnMinisom(x=3, y=3, sigma=1.0, learning_rate=0.5, num_iteration=1000)

        transformed = som.fit_transform(X)
        _transformed_with_y = som.fit_transform(X, y=[1, 2, 3, 4])

        self.assertEqual(transformed.shape, (X.shape[0], 2))

    def test_fit_predict(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = SklearnMinisom(
            x=3, y=3, sigma=1.0, learning_rate=0.5, num_iteration=1000, random_seed=42
        )

        predicted = som.fit_predict(X)

        self.assertTrue(all(predicted == predicted.astype(int)))
        self.assertEqual(predicted.shape, (X.shape[0],))

    def test_fit_predict_set_y(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = SklearnMinisom(
            x=3, y=3, sigma=1.0, learning_rate=0.5, num_iteration=1000, random_seed=42
        )

        predicted = som.fit_predict(X)
        predicted_with_y = som.fit_predict(X, y=[1, 2, 3, 4])

        self.assertTrue(all(predicted == predicted.astype(int)))
        self.assertEqual(predicted.any(), predicted_with_y.any())
        self.assertEqual(predicted.shape, (X.shape[0],))

    def test_set_params(self):
        som = SklearnMinisom(x=5, y=5, sigma=1.0, learning_rate=0.5, num_iteration=1000)
        som.set_params(sigma=0.8, learning_rate=0.3)
        self.assertEqual(som.sigma, 0.8)
        self.assertEqual(som.learning_rate, 0.3)
