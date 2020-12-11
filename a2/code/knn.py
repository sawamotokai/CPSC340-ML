"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils


class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the trianing data
        self.y = y

    def predict(self, Xtest):
        T, D = Xtest.shape
        N = self.X.shape[0]
        K = self.k
        labels = np.zeros((T, K))
        dists = utils.euclidean_dist_squared(self.X, Xtest)  # N by T
        sorted_args = np.argsort(dists, axis=0)

        yhat = np.zeros(T)
        for t in range(T):
            a = sorted_args[:K, t]
            yhat[t] = utils.mode(self.y[a])

        return yhat
