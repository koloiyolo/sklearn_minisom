"""
Author: Jakub Kołodziej
"""

from minisom import MiniSom
from numpy import array, linalg, ravel_multi_index
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn_minisom.parameters import (
    ActivationDistance,
    DecayFunction,
    NeighborhoodFunction,
    SigmaDecayFunction,
    Topology,
)


class SklearnMinisom(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        x: int = 10,
        y: int = 10,
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        num_iteration: int = 1000,
        decay_function: DecayFunction = DecayFunction.InverseDecayToZero,
        neighborhood_function: NeighborhoodFunction = NeighborhoodFunction.Gaussian,
        topology: Topology = Topology.Rectangular,
        activation_distance: ActivationDistance = ActivationDistance.Euclidean,
        random_seed: int | None = None,
        sigma_decay_function: SigmaDecayFunction = SigmaDecayFunction.InverseDecayToOne,
        random_order: bool = False,
        verbose: bool = False,
        use_epochs: bool = False,
        fixed_points: dict[int, tuple[int, int]] | None = None,
    ):
        """Initialize a MiniSom wrapper for use with scikit-learn.

        The estimator integrates with scikit-learn pipelines and tools such as
        ``GridSearchCV`` for hyperparameter optimization.

        Args:
            x (int): X dimension of the SOM. Defaults to 10.
            y (int): Y dimension of the SOM. Defaults to 10.
            sigma (float): Spread of the neighborhood function. It should be
                appropriate for the dimensions of the map and the neighborhood
                function. In some cases, it helps to set ``sigma`` to
                ``sqrt(x**2 + y**2)``. Defaults to 1.0.
            learning_rate (float): Initial learning rate. Appropriate values
                depend on the training data. By default, at iteration ``t``,
                ``learning_rate(t) = learning_rate / (1 + t * (100 / max_iter))``.
                Defaults to 0.5.
            num_iteration (int): Number of iterations. Appropriate values depend
                on the training data. Defaults to 1000.
            decay_function (DecayFunction): Function that reduces the learning
                rate at each iteration.
            neighborhood_function (NeighborhoodFunction): Function that weights
                the neighborhood of a position in the map.
            topology (Topology): Topology of the map.
            activation_distance (ActivationDistance): Distance used to activate
                the map.
            random_seed (int | None): Random seed to use. Defaults to None.
            sigma_decay_function (SigmaDecayFunction): Function that reduces
                ``sigma`` at each iteration. The default function is
                ``sigma(t) = sigma / (1 + (t * (sigma - 1) / max_iter))``.
                Defaults to ``SigmaDecayFunction.InverseDecayToOne``.
            random_order (bool): If True, samples in the SOM training function
                are picked in random order. Otherwise, samples are picked
                sequentially. Defaults to False.
            verbose (bool): If True, the training status is printed each time
                the weights are updated. Defaults to False.
            use_epochs (bool): If True, the SOM is trained for
                ``num_iteration`` epochs. In one epoch, the weights are updated
                ``len(data)`` times and the learning rate remains constant
                throughout the epoch. Defaults to False.
            fixed_points (dict[int, tuple[int, int]] | None): Mapping from a
                sample index ``k`` to neuron coordinates ``(c_1, c_2)``. The
                training algorithm uses the specified neuron as the winner for
                sample ``k`` instead of selecting the best matching unit.
                Defaults to None.
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
        """Fit the SOM to a data matrix.

        Args:
            X (np.array or list): Data matrix.
            y (Ignored): Not used; present for API consistency by convention.

        Returns:
            SklearnMinisom: This fitted estimator.

        Attributes:
            init_weights_ (ndarray of shape (grid_size_x, grid_size_y, feature_size)):
                Initial weights of the neural network.
            labels_ (ndarray of shape (n_samples,)):
                Labels of each point.
            weights_ (ndarray of shape (grid_size_x, grid_size_y, feature_size)):
                Weights of the neural network after training.
            n_features_in_ (int): Number of features seen during fit.
            inertia_ (float): Sum of squared distances of samples to their
                closest neuron weight vector, which provides a measure of the
                quality of the mapping.
        """
        if sparse.issparse(X):
            X = X.toarray()

        self.som = MiniSom(
            self.x,
            self.y,
            X.shape[1],
            sigma=self.sigma,
            learning_rate=self.learning_rate,
            decay_function=self.decay_function,
            neighborhood_function=self.neighborhood_function,
            topology=self.topology,
            activation_distance=self.activation_distance,
            random_seed=self.random_seed,
            sigma_decay_function=self.sigma_decay_function,
        )

        self.som.random_weights_init(X)
        self.init_weights_ = self.som.get_weights()
        self.som.train(
            X,
            self.num_iteration,
            random_order=self.random_order,
            verbose=self.verbose,
            use_epochs=self.use_epochs,
            fixed_points=self.fixed_points,
        )

        self.labels_ = self.predict(X)
        self.weights_ = self.som.get_weights()
        self.n_features_in_ = len(X[0])
        self.inertia_ = self._calculate_inertia(X=X)
        return self

    def transform(self, X):
        """Transform the data by finding the best matching unit (BMU) for each sample.

        Args:
            X (np.array or list): Data matrix.

        Returns:
            ndarray of shape (n_samples, 2): BMU coordinates for each input
                sample.
        """
        if sparse.issparse(X):
            X = X.toarray()
        return array([self.som.winner(x) for x in X])

    def predict(self, X):
        """Predict the cluster assignment (BMU) for each sample.

        The grid position (BMU) for each sample is returned as its cluster
        assignment. Each unique BMU is treated as a distinct cluster.

        Args:
            X (np.array or list): Data matrix.

        Returns:
            ndarray of shape (n_samples,): Index of the cluster each sample
                belongs to.
        """
        if sparse.issparse(X):
            X = X.toarray()
        bmu_coords = array([self.som.winner(x) for x in X])
        bmu_labels = ravel_multi_index(bmu_coords.T, (self.x, self.y))
        return bmu_labels

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the SOM and return the transformed BMU coordinates for each sample.

        This is a convenience method equivalent to calling ``fit(X)`` followed
        by ``transform(X)``.

        Args:
            X (np.array or list): Data matrix.
            y (Ignored): Not used; present for API consistency by convention.
            **fit_params (dict): Additional fit parameters. Currently ignored.

        Returns:
            ndarray of shape (n_samples, 2): BMU coordinates for each input
                sample.
        """
        if sparse.issparse(X):
            X = X.toarray()
        self.fit(X)
        return self.transform(X)

    def fit_predict(self, X, y=None):
        """Fit the SOM and return the predicted cluster assignments.

        This is a convenience method equivalent to calling ``fit(X)`` followed
        by ``predict(X)``.

        Args:
            X (np.array or list): Data matrix.
            y (Ignored): Not used; present for API consistency by convention.

        Returns:
            ndarray of shape (n_samples,): Index of the cluster each sample
                belongs to.
        """
        if sparse.issparse(X):
            X = X.toarray()
        self.fit(X)
        return self.predict(X)

    def get_params(self, deep: bool = True):
        """Get parameters of the estimator.

        This is helpful for hyperparameter tuning using ``GridSearchCV``.

        Args:
            deep (bool): If True, return the parameters for this estimator and
                contained subobjects that are estimators. Defaults to True.

        Returns:
            dict: Parameter names mapped to their values.
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
            "fixed_points": self.fixed_points,
        }

    def set_params(self, **params):
        """Set the parameters of this estimator.

        This allows setting parameters during ``GridSearchCV``. The method
        works on simple estimators as well as nested objects, such as
        ``sklearn.pipeline.Pipeline``. Nested parameters have the form
        ``<component>__<parameter>`` so that each component can be updated.

        Args:
            **params (dict): Estimator parameters.

        Returns:
            estimator instance: This estimator.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def _calculate_inertia(self, X):
        """Measure cluster compactness in the context of a SOM.

        Args:
            X (np.array): Data matrix.

        Returns:
            float: Inertia score based on distances to the nearest SOM neuron.
        """
        inertia = 0
        for i in range(X.shape[0]):
            bmu = self.som.winner(X[i])
            distance = linalg.norm(X[i] - self.som.get_weights()[bmu[0], bmu[1]])
            inertia += distance**2

        return inertia / X.shape[0]
