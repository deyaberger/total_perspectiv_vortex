import numpy as np
from scipy import linalg
from sklearn.base import TransformerMixin, BaseEstimator

class My_CSP(TransformerMixin, BaseEstimator):
    def __init__(self, n_components: int = 4):
        if not isinstance(n_components, int):
            raise ValueError('n_components must be an integer.')
        self.n_components = n_components

    def _compute_covariance_matrices(self, X, y):
        events, channels, time_stamps = X.shape

        covariances = []
        for current_class in self._classes:
            """Concatenate epochs before computing the covariance."""
            x_class = X[y==current_class]
            x_class = np.transpose(x_class, [1, 0, 2])
            x_class = x_class.reshape(channels, -1)
            cov_mat = np.cov(x_class)
            covariances.append(cov_mat)

        return np.stack(covariances)

    def fit(self, X, y):
        """Estimate the CSP decomposition on epochs."""

        self._classes = np.unique(y)

        covariances = self._compute_covariance_matrices(X, y)
        eigen_values, eigen_vectors = linalg.eigh(covariances[0], covariances.sum(0))

        ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        eigen_vectors = eigen_vectors[:, ix]

        self.filters_ = eigen_vectors.T
        pick_filters = self.filters_[:self.n_components]

        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        X = (X ** 2).mean(axis=2)

        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)." % type(X))
        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean band power)
        X = (X ** 2).mean(axis=2)
        # X = np.log(X)
        X -= X.mean()
        X /= X.std()
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
