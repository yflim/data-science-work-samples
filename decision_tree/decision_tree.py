# Based loosely on https://github.com/SebastianMantey/Random-Forest-from-Scratch

import numpy as np
import pandas as pd
import operator
import random


# TODO: ***COST COMPLEXITY PRUNING***

def is_numeric(var, int_as_numeric=True):
    """
    Determines whether variable is numeric
    :param var: Obesrvations of single variable
    :type var: Union[array, Series]
    :return: (bool) Is variable numeric?
    """
    return (int_as_numeric and (var.dtype.kind in np.typecodes["AllInteger"])) or (var.dtype.kind in np.typecodes["AllFloat"])

def get_comparator(var):
    """
    Returns appropriate comparator depending on variable type (<= for numeric, == for categorical)
    :param var: Obesrvations of single variable
    :type var: Union[array, Series]
    :return: (builtin_function_or_method) Python operator
    """
    if is_numeric(var):
        return operator.le
    return operator.eq


# Binary decision tree
class DecisionTree():
    def __init__(self, x, y, max_features=None, max_depth=None, min_samples=2):
        """
        Constructor for classification / regression Decision Tree
        :param x: (DataFrame) Training feature data
        :param y: Training dependent variable data
        :param max_features: (int) Number of features to consider per split
        :param max_depth: (int) maximum number of subbranches
        :param min_samples: (int) Minimum number of samples per branch / leaf node
        """
        self.features = x.columns
        self.data = x
        self.feature_comparators = self.data.apply(lambda feature: get_comparator(feature))
        self.data['dependent'] = y
        self.is_classifier = not(is_numeric(self.data.dependent, False))
        self.max_features = max_features
        self.max_depth = max_depth or (self.max_features and (2 * self.max_features))
        self.min_samples = min_samples
        self.tree_dict = {}

    def _is_pure(self, obs_index):
        """
        Checks whether a node is a leaf due to homogeneous dependent variable value
        :param obs_index: (Index) DataFrame indices of observations to be considered
        :return: (bool) Is a leaf?
        """
        return len(np.unique(self.data.dependent[obs_index])) == 1

    def _assign_leaf_val(self, obs_index):
        """
        Assign the value associated with a leaf
        :param obs_index: (Index) DataFrame indices of observations in leaf
        :return: (object) Predicted dependent variable value (mean if numeric; mode if categorical)
        """
        y = self.data.dependent[obs_index]
        if self.is_classifier:
            unique_classes, counts_unique_classes = np.unique(y, return_counts=True)
            return unique_classes[counts_unique_classes.argmax()]
        return y.mean()

    def _get_potential_splits(self, obs_index):
        """
        Return all potential splits of the indexed subset of data
        :param obs_index: (Index) DataFrame indices of feature observations to be considered
        :return: (dict) Possible splits
        """
        x = self.data.loc[obs_index, self.features]
        cols = self.features
        if self.max_features:
            column_indices = random.sample(population=list(range(len(self.features))), k=self.max_features)
            cols = cols[column_indices]
        return { feature: np.unique(x[feature]) for feature in cols }

    def _split_data(self, obs_index, split_column, split_value):
        """
        Splits the data
        :param split_column: Column to split the data on
        :param split_value: Value to split on
        :return index_below, index_above: (Index, Index) DataFrame indices of observations in each partition
        """
        data, op = self.data.loc[obs_index], self.feature_comparators[split_column]
        index_below = data[op(data[split_column], split_value)].index
        index_above = data.index.difference(index_below)
        return index_below, index_above

    def _get_entropy(self, obs_index):
        """"
        Calculates entropy of a set of data
        :param obs_index: (Index) DataFrame indices of observations to be considered
        :returns: (float) Entropy
        """
        _, counts = np.unique(self.data.loc[obs_index].dependent, return_counts=True)
        probabilities = counts * 1.0 / counts.sum()
        return -(probabilities * np.log2(probabilities)).sum()

    def _get_rss(self, obs_index):
        """"
        Calculates RSS of a set of data
        :param obs_index: (Index) DataFrame indices of observations to be considered
        :returns: (float) RSS
        """
        y_mean = self._assign_leaf_val(obs_index)
        return np.square(self.data.loc[obs_index].dependent - y_mean).sum()

    def _get_split_entropy(self, index_below, index_above):
        """
        Calculates the entropy of a split
        :param index_below: (Index) DataFrame indices of data in lower/equal branch
        :param index_above: (Index) DataFrame indices of data in higher/unequal branch
        :return: (float) Entropy of split
        """
        n = (len(index_below) + len(index_above)) * 1.0
        fract_below, fract_above = len(index_below) / n, len(index_above) / n
        return fract_below * self._get_entropy(index_below) + fract_above * self._get_entropy(index_above)

    def _get_split_rss(self, index_below, index_above):
        """
        Calculates the RSS of a split
        :param index_below: (Index) DataFrame indices of data in lower/equal branch
        :param index_above: (Index) DataFrame indices of data in higher/unequal branch
        :return: (float) RSS of split
        """
        return self._get_rss(index_below) + self._get_rss(index_above)

    def _get_split_cost(self, index_below, index_above):
        """
        Calculates the cost of a split
        :param index_below: (Index) DataFrame indices of data in lower/equal branch
        :param index_above: (Index) DataFrame indices of data in higher/unequal branch
        :return: (float) Cost of split
        """
        if self.is_classifier:
            return self._get_split_entropy(index_below, index_above)
        return self._get_split_rss(index_below, index_above)

    def _determine_best_split(self, obs_index, potential_splits):
        """
        Determines the best split out of the potential splits via an exhaustive search
        :param potential_splits: (dict) Potential splits to consider
        :return best_split_column, best_split_value: (object, float) The best split
        """
        best_split_cost = np.inf
        for feature in potential_splits.keys():
            for value in potential_splits[feature]:
                index_below, index_above = self._split_data(obs_index, feature, value)
                current_split_cost = self._get_split_cost(index_below, index_above)
                if current_split_cost <= best_split_cost:
                    best_split_cost = current_split_cost
                    best_split_column = feature
                    best_split_value = value
        return best_split_column, best_split_value

    def _grow(self, obs_index, counter=0, branches_must_differ=False):
        """
        Adds a branch to decision tree dictionary (self.tree_dict)
        :param obs_index: absolute indices of observations to be considered
        :param counter: (int) Current depth of branch
        """
        data = self.data.loc[obs_index]
        if (counter == self.max_depth) or (self.is_classifier and self._is_pure(obs_index)):
            return (self._assign_leaf_val(obs_index), obs_index)
        else:
            counter += 1
            potential_splits = self._get_potential_splits(obs_index)
            split_column, split_value = self._determine_best_split(obs_index, potential_splits)
            index_below, index_above = self._split_data(obs_index, split_column, split_value)
            if len(index_below) < self.min_samples or len(index_above) < self.min_samples:
                return (self._assign_leaf_val(obs_index), obs_index)
            yes_answer = self._grow(index_below, counter=counter)
            no_answer = self._grow(index_above, counter=counter)
            # In a random forest tree, if the answers are the same, then there is no point in asking the question.
            # This could happen when the data is classified even though it is not pure yet.
            if branches_must_differ and (not isinstance(yes_answer, dict)) and (not isinstance(no_answer, dict)):
                if (yes_answer[0] == no_answer[0]):
                    return (yes_answer[0], obs_index)
            return { (split_column, split_value): [yes_answer, no_answer] }

    def fit(self, branches_must_differ=False):
        """
        Fits a decision tree to the data used
        :param branches_must_differ: (bool) Stop splitting if child branches share predicted value
        """
        self.tree_dict = self._grow(self.data.index, 0, branches_must_differ)

    def predict_observation(self, observation, tree):
        """
        Predicts the label of a single observation using given subtree
        :param tree: (dict) Subtree to use
        :param observation: (Series) Single feature vector
        :return label: (object) Predicted class
        """
        key = list(tree.keys())[0]
        feature, val = key[0], key[1]
        op = self.feature_comparators[feature]
        answer = tree[key][int(not(op(observation[feature], val)))]
        if not isinstance(answer, dict):
            return answer
        return self.predict_observation(observation, answer)

    def predict(self, x):
        """
        Get predictions for whole dataset
        :param x: (DataFrame) Feature data to predict on
        :return: (Series) Predictions
        """
        return x.apply(self.predict_observation, args=(self.tree_dict,), axis=1)

    def _grow(self, obs_index, counter=0, branches_must_differ=False):
        """
        Adds a branch to decision tree dictionary (self.tree_dict)
        :param obs_index: absolute indices of observations to be considered
        :param counter: (int) Current depth of branch
        """
        data = self.data.loc[obs_index]
        if (counter == self.max_depth) or (self.is_classifier and self._is_pure(obs_index)):
            return self._assign_leaf_val(obs_index)
        else:
            counter += 1
            potential_splits = self._get_potential_splits(obs_index)
            split_column, split_value = self._determine_best_split(potential_splits)
            index_below, index_above = self._split_data(split_column, split_value)
            if len(index_below) < self.min_samples or len(index_above) < self.min_samples:
                return self._assign_leaf_val(obs_index)
            yes_answer = self._grow(index_below, counter=counter)
            no_answer = self._grow(index_above, counter=counter)
            # In a random forest tree, if the answers are the same, then there is no point in asking the question.
            # This could happen when the data is classified even though it is not pure yet.
            if branches_must_differ and (yes_answer == no_answer):
                return yes_answer
            return { (split_column, split_value): [yes_answer, no_answer] }

    def fit(self, branches_must_differ=False):
        self.tree_dict = self._grow(self.data.index, 0, branches_must_differ)

