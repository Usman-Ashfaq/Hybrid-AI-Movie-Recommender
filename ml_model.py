import numpy as np
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor

class MLModel:
    def __init__(self, movies):
        self.movies = movies

    def prepare(self):
        X = []
        y = []

        for m in self.movies:
            X.append([
                int(m['year']),
                float(m['runtime']),
                float(m['popularity'])
            ])
            y.append(float(m['vote_average']))

        return np.array(X), np.array(y)

    def train(self):
        X, y = self.prepare()

        self.kmeans = KMeans(n_clusters=3, random_state=0)
        self.kmeans.fit(X)

        self.ann = MLPRegressor(hidden_layer_sizes=(20, 10), max_iter=500)
        self.ann.fit(X, y)

    def predict(self, m):
        X = np.array([[int(m['year']), float(m['runtime']), float(m['popularity'])]])
        return self.ann.predict(X)[0]

    def get_cluster(self, m):
        X = np.array([[int(m['year']), float(m['runtime']), float(m['popularity'])]])
        return self.kmeans.predict(X)[0]