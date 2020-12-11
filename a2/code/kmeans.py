import numpy as np
from utils import euclidean_dist_squared


class Kmeans:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        # pick a random points as means
        means = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            means[kk] = X[i]
        self.means = means

        while True:
            y_old = y

            # Compute euclidean distance to each mean
            dist2 = euclidean_dist_squared(X, means)
            dist2[np.isnan(dist2)] = np.inf
            # Yi = Xi's temporary closest mean
            y = np.argmin(dist2, axis=1)

            # Update means
            for kk in range(self.k):
                # don't update the mean if no examples are assigned to it (one of several possible approaches)
                if np.any(y == kk):
                    means[kk] = X[y == kk].mean(axis=0)

            changes = np.sum(y != y_old)
            # print('Running K-means, changes in cluster assignment = {}'.format(changes))
            print(self.error(X))
            # Stop if no point changed cluster
            if changes == 0:
                break

        self.means = means

    def predict(self, X):
        means = self.means
        dist2 = euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1)

    def error(self, X):
        means = self.means
        dist2 = euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        closest_means = np.argmin(dist2, axis=1)
        N = X.shape[0]
        sum_dist = 0
        for n in range(N):
            c = closest_means[n]
            sum_dist += dist2[n, c]
        return sum_dist
