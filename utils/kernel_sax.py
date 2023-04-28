import numpy as np


class KernelSAX:
    def __init__(self, n_samples=1000, n_bins=8, alpha=0.3, sigma=1):
        self.n_samples = n_samples
        self.n_bins = n_bins
        self.alpha = alpha
        self.sigma = sigma

    def _gaussian_kernel(self, x, y):
        return np.exp(-np.sum((x - y) ** 2) / (2 * self.sigma**2))

    def _get_word(self, x, centroids):
        distances = [self._gaussian_kernel(x, c) for c in centroids]
        return np.argmax(distances)

    def _get_centroids(self, data):
        n_data = data.shape[0]
        idx = np.random.choice(n_data, self.n_samples)
        centroids = data[idx]
        for i in range(10):
            distances = np.zeros((self.n_samples, n_data))
            for j in range(self.n_samples):
                for k in range(n_data):
                    distances[j, k] = self._gaussian_kernel(centroids[j], data[k])
            weights = np.exp(-self.alpha * distances)
            weights /= np.sum(weights, axis=0)
            centroids = np.dot(weights, data) / np.sum(weights, axis=1)[:, None]
        return centroids

    def fit_transform(self, data):
        data = np.array(data)
        centroids = self._get_centroids(data)
        words = np.zeros((data.shape[0],))
        for i in range(data.shape[0]):
            words[i] = self._get_word(data[i], centroids)
        codewords = np.zeros((data.shape[0], self.n_bins))
        for i in range(self.n_bins):
            codewords[:, i] = np.sum(words == i, axis=1)
        codewords /= np.sum(codewords, axis=1)[:, None]
        return codewords
