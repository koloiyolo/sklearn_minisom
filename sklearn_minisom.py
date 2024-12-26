"""
Author: Jakub Kołodziej
"""

from minisom import MiniSom
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
from numpy import array, ravel_multi_index, all, linalg

# for unit tests
import unittest


class MiniSOM(BaseEstimator, TransformerMixin):
    def __init__(self, x=10, y=10, sigma=1.0, learning_rate=0.5,
                 num_iteration=1000, decay_function='asymptotic_decay',
                 neighborhood_function='gaussian', topology='rectangular',
                 activation_distance='euclidean', random_seed=None,
                 sigma_decay_function='asymptotic_decay',
                 random_order=False, verbose=False,
                 use_epochs=False, fixed_points=None):
        """
        Minisom wrapper that integrates seamlessly with the
        Scikit-learn ecosystem, providing a familiar API for users.

        It enables easy integration with Scikit-learn pipelines and
        tools like GridSearchCV for hyperparameter optimization.

        Parameters
        ----------
        x : int
            x dimension of the SOM.

        y : int
            y dimension of the SOM.

        sigma : float, optional (default=1)
            Spread of the neighborhood function.

            Needs to be adequate to the dimensions of the map
            and the neighborhood function. In some cases it
            helps to set sigma as sqrt(x^2 +y^2).

        learning_rate : float, optional (default=0.5)
            Initial learning rate.

            Adequate values are dependent on the data used for training.

            By default, at the iteration t, we have:
                learning_rate(t) = learning_rate / (1 + t * (100 / max_iter))

        num_iteration : int, optional (default=1000)
            Number of iterations.

            Adequate values are dependent on the data used for training.

        decay_function : string or callable, optional
        (default='inverse_decay_to_zero')
            Function that reduces learning_rate at each iteration.
            Possible values: 'inverse_decay_to_zero', 'linear_decay_to_zero',
                             'asymptotic_decay' or callable

            If a custom decay function using a callable
            it will need to to take in input
            three parameters in the following order:

            1. learning rate
            2. current iteration
            3. maximum number of iterations allowed

            Note that if a lambda function is used to define the decay
            MiniSom will not be pickable anymore.

        neighborhood_function : string, optional (default='gaussian')
            Function that weights the neighborhood of a position in the map.
            Possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'

        topology : string, optional (default='rectangular')
            Topology of the map.
            Possible values: 'rectangular', 'hexagonal'

        activation_distance : string, callable optional (default='euclidean')
            Distance used to activate the map.
            Possible values: 'euclidean', 'cosine', 'manhattan', 'chebyshev'

            Example of callable that can be passed:

            def euclidean(x, w):
                return linalg.norm(subtract(x, w), axis=-1)

        random_seed : int, optional (default=None)
            Random seed to use.

        sigma_decay_function : string, optional
        (default='inverse_decay_to_one')
            Function that reduces sigma at each iteration.
            Possible values: 'inverse_decay_to_one', 'linear_decay_to_one',
                             'asymptotic_decay'

            The default function is:
                sigma(t) = sigma / (1 + (t * (sigma - 1) / max_iter))

        random_order : bool (default=False)
            If True, samples in SOM train function are picked in random order.
            Otherwise the samples are picked sequentially.

        verbose : bool (default=False)
            If True the status of the training will be
            printed each time the weights are updated.

        use_epochs : bool (default=False)
            If True the SOM will be trained for num_iteration epochs.
            In one epoch the weights are updated len(data) times and
            the learning rate is constat throughout a single epoch.

        fixed_points : dict (default=None)
            A dictionary k : (c_1, c_2), that will force the
            training algorithm to use the neuron with coordinates
            (c_1, c_2) as winner for the sample k instead of
            the best matching unit.
        """

        self.x = x
        self.y = y
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.decay_function = decay_function
        self.neighborhood_function = neighborhood_function
        self.topology = topology
        self.activation_distance = activation_distance
        self.random_seed = random_seed
        self.sigma_decay_function = sigma_decay_function
        self.random_order = random_order
        self.verbose = verbose
        self.use_epochs = use_epochs
        self.fixed_points = fixed_points

    def fit(self, X, y=None):
        """
        Initializes SOM algorithm from minisom library
        and fits on it data matrix.

        Parameters
        ----------
        X : np.array or list
            Data matrix.

        y : Ignored
            Not used, present here for API consistency by convention.
        """
        if sparse.issparse(X):
            X = X.toarray()

        self.som = MiniSom(self.x, self.y, X.shape[1],
                           sigma=self.sigma,
                           learning_rate=self.learning_rate,
                           decay_function=self.decay_function,
                           neighborhood_function=self.neighborhood_function,
                           topology=self.topology,
                           activation_distance=self.activation_distance,
                           random_seed=self.random_seed,
                           sigma_decay_function=self.sigma_decay_function)

        self.som.random_weights_init(X)
        self.init_weights_ = self.som.get_weights()
        """
        labels_ : ndarray of shape (n_samples,)

        Returns the initial weights of the neural network.
        """
        self.som.train(X, self.num_iteration,
                       random_order=self.random_order,
                       verbose=self.verbose,
                       use_epochs=self.use_epochs,
                       fixed_points=self.fixed_points)

        self.labels_ = self.predict(X)
        """
        labels_ : ndarray of shape (n_samples,)

        Labels of each point.
        """
        self.weights_ = self.som.get_weights()
        """
        weights_ : ndarray of shape (grid_size_x, grid_size_y, feature_size)

        Returns the weights of the neural network.
        """
        self.n_features_in_ = len(X[0])
        """
        n_features_in_ : int

        Number of features seen during fit.
        """
        self.inertia_ = self._calculate_inertia(X=X)
        """
        inertia_ : float

        Sum of squared distances of samples to their closest neuron weight
        vector, which provides a measure of the quality of the mapping.
        """
        return self

    def transform(self, X):
        """
        Transform the data by finding the best matching unit (BMU) for
        each sample. Returns the BMU coordinates for each input sample.

        Parameters
        ----------
        X : np.array or list
            Data matrix.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        if sparse.issparse(X):
            X = X.toarray()
        return array([self.som.winner(x) for x in X])

    def predict(self, X):
        """
        Predict the cluster assignment (BMU) for each sample.
        Returns the grid position (BMU) for each sample as the
        cluster assignment. Treats each unique BMU as a distinct cluster.

        Parameters
        ----------
        X : np.array or list
            Data matrix.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if sparse.issparse(X):
            X = X.toarray()
        bmu_coords = array([self.som.winner(x) for x in X])
        bmu_labels = ravel_multi_index(bmu_coords.T, (self.x, self.y))
        return bmu_labels

    def fit_transform(self, X, y=None):
        """
        Fit the SOM and return the transformed
        BMU coordinates for each input sample.

        Convenience method; equivalent to calling fit(X) followed by
        transform(X).

        Parameters
        ----------
        X : np.array or list
            Data matrix.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        if sparse.issparse(X):
            X = X.toarray()
        self.fit(X)
        return self.transform(X)

    def fit_predict(self, X, y=None):
        """
        Fit the SOM and return the predicted cluster assignments.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : np.array or list
            Data matrix.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if sparse.issparse(X):
            X = X.toarray()
        self.fit(X)
        return self.predict(X)

    def get_params(self, deep=True):
        """
        Get parameters of the estimator.

        Helpful for hyperparameter tuning using GridSearchCV.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "x": self.x,
            "y": self.y,
            "sigma": self.sigma,
            "learning_rate": self.learning_rate,
            "num_iteration": self.num_iteration,
            "decay_function": self.decay_function,
            "neighborhood_function": self.neighborhood_function,
            "topology": self.topology,
            "activation_distance": self.activation_distance,
            "random_seed": self.random_seed,
            "sigma_decay_function": self.sigma_decay_function,
            "random_order": self.random_order,
            "verbose": self.verbose,
            "use_epochs": self.use_epochs,
            "fixed_points": self.fixed_points
        }

    def set_params(self, **params):

        """
        Set the parameters of this estimator.
        This allows setting parameters during GridSearchCV.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def _calculate_inertia(self, X):
        """
        Custom inertia function to measure the compactness of clusters
        in the context of Self-Organizing Maps (SOM).

        Parameters
        ----------
        X : np.array
            Data matrix.

        Returns
        -------
        float
            The inertia score based on distances to nearest SOM neuron.
        """
        inertia = 0
        for i in range(X.shape[0]):
            bmu = self.som.winner(X[i])
            distance = linalg.norm(X[i] - self.som.get_weights()[bmu[0],
                                   bmu[1]])
            inertia += distance ** 2

        return inertia / X.shape[0]


class TestMinisom(unittest.TestCase):
    def test_initialization(self):
        som = MiniSOM(x=2, y=10, sigma=2.0, learning_rate=0.7,
                      num_iteration=100, neighborhood_function='test1',
                      topology='test2', activation_distance='test3')

        self.assertEqual(som.x, 2)
        self.assertEqual(som.y, 10)
        self.assertEqual(som.sigma, 2.0)
        self.assertEqual(som.learning_rate, 0.7)
        self.assertEqual(som.num_iteration, 100)
        self.assertEqual(som.neighborhood_function, 'test1')
        self.assertEqual(som.topology, 'test2')
        self.assertEqual(som.activation_distance, 'test3')

    def test_fit(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = MiniSOM(x=3, y=3, sigma=1.0,
                      learning_rate=0.5,
                      num_iteration=1000)
        som.fit(X)

        self.assertIsNotNone(som.som)

    def test_fit(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = MiniSOM(x=3, y=3, sigma=1.0,
                      learning_rate=0.5,
                      num_iteration=1000)
        som.fit(X)

        self.assertIsNotNone(som.som)

    def test_transform(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = MiniSOM(x=3, y=3, sigma=1.0,
                      learning_rate=0.5,
                      num_iteration=1000)
        som.fit(X)
        transformed = som.transform(X)

        self.assertEqual(transformed.shape, (X.shape[0], 2))

    def test_predict(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = MiniSOM(x=3, y=3, sigma=1.0,
                      learning_rate=0.5,
                      num_iteration=1000)
        som.fit(X, y=[1, 2, 3, 4])
        predicted = som.predict(X)

        self.assertTrue(all(predicted == predicted.astype(int)))
        self.assertEqual(predicted.shape, (X.shape[0],))

    def test_fit_transform(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = MiniSOM(x=3, y=3, sigma=1.0,
                      learning_rate=0.5,
                      num_iteration=1000)
        transformed = som.fit_transform(X)

        self.assertEqual(transformed.shape, (X.shape[0], 2))

    def test_fit_transform_set_y(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = MiniSOM(x=3, y=3, sigma=1.0,
                      learning_rate=0.5,
                      num_iteration=1000)

        transformed = som.fit_transform(X)
        transformed_with_y = som.fit_transform(X, y=[1, 2, 3, 4])

        self.assertEqual(transformed.shape, (X.shape[0], 2))

    def test_fit_predict(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = MiniSOM(x=3, y=3, sigma=1.0,
                      learning_rate=0.5,
                      num_iteration=1000,
                      random_seed=42)

        predicted = som.fit_predict(X)

        self.assertTrue(all(predicted == predicted.astype(int)))
        self.assertEqual(predicted.shape, (X.shape[0],))

    def test_fit_predict_set_y(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = MiniSOM(x=3, y=3, sigma=1.0,
                      learning_rate=0.5,
                      num_iteration=1000,
                      random_seed=42)

        predicted = som.fit_predict(X)
        predicted_with_y = som.fit_predict(X, y=[1, 2, 3, 4])

        self.assertTrue(all(predicted == predicted.astype(int)))
        self.assertEqual(predicted.any(), predicted_with_y.any())
        self.assertEqual(predicted.shape, (X.shape[0],))

    def test_set_params(self):
        som = MiniSOM(x=5, y=5, sigma=1.0,
                      learning_rate=0.5,
                      num_iteration=1000)
        som.set_params(sigma=0.8, learning_rate=0.3)
        self.assertEqual(som.sigma, 0.8)
        self.assertEqual(som.learning_rate, 0.3)
