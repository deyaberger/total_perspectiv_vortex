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
        """Find the CSP decomposition of epochs.

        1. Calulate Covariance Matrix: a square matrix denoting the covariance of the elements with each other
        2. Compute Eignenvalues and Eigenvectors: they represent magnitude, or importance.
            Bigger Eigenvalues correlate with more important directions.
        3. Select Higher Eigenvalues: if n_components = 10: we select the 10 biggest eigen values
        """

        self._classes = np.unique(y)

        # The covariance matrix is a square matrix denoting the covariance of the elements with each other. The covariance of an element with itself is nothing but just its Variance.
        covariances = self._compute_covariance_matrices(X, y)
        #  The Eigenvectors of the Covariance matrix we get are Orthogonal to each other and each vector represents a principal axis.
        eigen_values, eigen_vectors = linalg.eigh(covariances[0], covariances.sum(0))
        # A Higher Eigenvalue corresponds to a higher variability. Hence the principal axis with the higher Eigenvalue will be an axis capturing higher variability in the data.

        ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        eigen_vectors = eigen_vectors[:, ix]

        self.filters_ = eigen_vectors.T
        pick_filters = self.filters_[:self.n_components]

        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        X = (X ** 2).mean(axis=2)

        return self

    def transform(self, X):
        """Get Data reduce to lower dimensions.

        To transform the data, we need to do a dot product between the Transpose of the Eigenvector subset
        and the Transpose of the mean-centered data.
        - pick_filter: Eigenvector subset
        - epoch: mean-centered data
        """
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
        """Fit then transform."""
        self.fit(X, y)
        return self.transform(X)
