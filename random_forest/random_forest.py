# Based on https://github.com/SebastianMantey/Random-Forest-from-Scratch
# NOTE: Docstrings not completely accurate about type, e.g. some methods can handle both np.ndarray and Series
# WARNING: FULLY TESTED YET.

import decision_tree
import math as m
import numpy as np
import pandas as pd
import random


def bootstrap(data, n_bootstrap):
    """
    Bootstrapped sample selection
    :param data: (DataFrame) Dataset to bootstrap on
    :param n_bootstrap: (int) number of samples to select
    :return:
    """
    return data.iloc[np.random.randint(low=0, high=data.shape[0]-1, size=n_bootstrap)]

class RandomForest():
    def __init__(self, n_trees=200, n_bootstrap=None, max_features=None, max_depth=None, min_samples=2):
        """
        Constructor for Random Forest classifier / regressor
        :param n_trees: (int) Number of trees to generate
        :param n_bootstrap: (int) number of samples to bootstrap
        :param max_features: (int) Number of features to consider per split
        :param max_depth: (int) Maximum number of subbranches
        :param min_samples: (int) Minimum number of samples per branch / leaf node
        """
        self.n_trees, self.n_bootstrap, self.max_features, self.max_depth, self.min_samples = n_trees, n_bootstrap, max_features, max_depth, min_samples

    def fit(self, x, y):
        """
        Fit the model on the data
        :param x: (Dataframe) Feature data
        :param y: (array) Dependent variable
        """
        if not self.n_bootstrap:
            self.n_bootstrap = ((len(x) <= 1000) and min(250, len(x))) or 500
        features = x.columns
        if not self.max_features:
            self.max_features = m.ceil(len(features) ** 0.5)
        data = x
        data['dependent'] = y
        self.forest = []
        for i in range(self.n_trees):
            data_bs = bootstrap(data, self.n_bootstrap)
            self.forest[i] = decision_tree.DecisionTree(data_bs[features], data_bs.dependent,
                max_features=self.max_features, max_depth=self.max_depth, min_samples=self.min_samples)
            self.forest[i].grow(True)

    def predict(self, x):
        """
        Predict
        :param x: (Dataframe) Data to predict on
        :return: (Series) Predictions
        """
        df_predictions = pd.DataFrame({ i: self.forest[i].predict(x) for i in range(len(self.forest)) })
        if is_numeric(df_predictions[0]):
            return df_predictions.mean(axis=1)[0]
        return df_predictions.mode(axis=1)[0]

