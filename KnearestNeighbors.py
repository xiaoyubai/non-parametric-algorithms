from sklearn.datasets import make_classification
import numpy as np
from numpy.linalg import norm
import sklearn.metrics as skm
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn import neighbors
from matplotlib.colors import ListedColormap



X, y = make_classification(n_features=4, n_redundant=0, n_informative=1,
                           n_clusters_per_class=1, class_sep=5, random_state=5)

train_X, test_X, train_y, test_y = train_test_split(X, y)

def euclidean_distance(n1, n2):
    return np.sqrt(np.sum((n1 - n2) ** 2))

def cosine_distance(n1, n2):
    return 1 - np.dot(n1, n2) / (norm(n1) * norm(n2))

class KnearestNeighbors(object):
    def __init__(self, k, distance):
        self.k = k
        self.distance = distance

    def fit(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y

    def predict(self, test_X):
        y_preds = []
        for x_test in test_X:
            distances = []
            for x_train in self.train_X:
                distances.append(self.distance(x_train, x_test))
            #distances = sorted(distances)[:k]
            distances = np.array(distances)
            sortedDistIndices = distances.argsort()
            y_pred = np.round(np.sum(self.train_y[sortedDistIndices < self.k]) / float(self.k))
            y_preds.append(y_pred)
        return y_preds

    def score(self, y_preds, y_test):
        return skm.accuracy_score(y_preds, y_test)

knn = KnearestNeighbors(k=3, distance=euclidean_distance)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
h = .5
X = X[:, :2]
knn.fit(X,y)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.array(Z)
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i, weights = '%s')" % (3, weights))
pltl.show()
