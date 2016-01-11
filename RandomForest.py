from DecisionTree import DecisionTree
import numpy as np
from scipy.stats import mode

class RandomForest(object):
    '''A Random Forest class'''

    def __init__(self, num_trees, num_features):
        '''
           num_trees:  number of trees to create in the forest:
        num_features:  the number of features to consider when choosing the
                           best split for each node of the decision trees
        '''
        self.num_trees = num_trees
        self.num_features = num_features
        self.forest = None

    def fit(self, X, y):
        '''
        X:  two dimensional numpy array representing feature matrix
                for test data
        y:  numpy array representing labels for test data
        '''
        self.forest = self.build_forest(X, y, self.num_trees, X.shape[0], \
                                        self.num_features)

    def build_forest(self, X, y, num_trees, num_samples, num_features):
        '''
        Return a list of num_trees DecisionTrees.
        '''
        size = len(y)
        index = range(size)
        trees = []
        for tree in range(num_trees):
            random_sample_index = np.random.choice(index, size, replace=True)
            X_random = X[random_sample_index]
            y_random = y[random_sample_index]
            dt = DecisionTree(num_features)
            dt.fit(X_random, y_random)
            trees.append(dt)
        return trees

    def predict(self, X):
        '''
        Return a numpy array of the labels predicted for the given test data.
        '''
        predictions = []
        for tree in self.forest:
            predicted_y = tree.predict(X)
            predictions.append(predicted_y)
        predictions = np.array(predictions)
        return mode(predictions, axis=0)[0][0]




    def score(self, X, y):
        '''
        Return the accuracy of the Random Forest for the given test data and
        labels.
        '''
        mask = np.ones(len(y))
        pred = self.predict(X)
        y_choice = np.unique(y)
        y_1choice = y_choice[0]
        TP = sum(mask[(y==y_choice[0]) & (pred==y_choice[0])])
        TN = sum(mask[(y!=y_choice[0]) & (pred!=y_choice[0])])
        return (TP + TN) / float(len(y))
