# Based loosely on https://github.com/SebastianMantey/Random-Forest-from-Scratch

import cross_validation.cv as cv
import numpy as np
import pandas as pd
import operator
import random
import utils


def get_comparator(var):
    """
    Returns appropriate comparator depending on variable type (<= for numeric, == for categorical)
    :param var: Obesrvations of single variable
    :type var: Union[array, Series]
    :return: (builtin_function_or_method) Python operator
    """
    if utils.is_numeric(var):
        return operator.le
    return operator.eq

# Binary decision tree
class DecisionTree():
    def __init__(self, x, y, max_features=None, max_depth=None, min_samples=None):
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
        self.is_classifier = not(utils.is_numeric(self.data.dependent, False))
        self.max_features = max_features
        self.max_depth = max_depth or (self.max_features and (2 * self.max_features))
        self.min_samples = min_samples or max(1, round(0.001 * len(self.data)))
        self.tree_dict = {}
        self.pruned = False

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
            return answer[0]
        return self.predict_observation(observation, answer)

    def predict(self, x, subtree=None):
        """
        Get predictions for given feature data
        :param x: (DataFrame) Feature data to predict on
        :return: (Series) Predictions
        """
        subtree = subtree or self.tree_dict
        if not subtree:
            self.fit()
        if not isinstance(subtree, dict):
            return subtree[0]
        return x.apply(self.predict_observation, args=(subtree,), axis=1)

    def _get_min_cost_subtree(self, tree, alpha=0):
        cost_method = self._get_rss
        if self.is_classifier:
            cost_method = self._get_entropy
        if not isinstance(tree, dict):
            return (tree, tree[1], cost_method(tree[1]), alpha)
        branches = list(tree.values())[0]
        min_cost_subtree_l = self._get_min_cost_subtree(branches[0], alpha)
        min_cost_subtree_r = self._get_min_cost_subtree(branches[1], alpha)
        obs_index = min_cost_subtree_l[1].union(min_cost_subtree_r[1])
        leaf_cost = cost_method(obs_index)
        branching_cost = min_cost_subtree_l[2] + min_cost_subtree_r[2]
        branching_alpha = min_cost_subtree_l[3] + min_cost_subtree_r[3]
        if self.is_classifier:
            wt_l = len(min_cost_subtree_l[1]) * 1.0 / len(obs_index)
            wt_r = len(min_cost_subtree_r[1]) * 1.0 / len(obs_index)
            branching_cost = (wt_l * min_cost_subtree_l[2]) + (wt_r * min_cost_subtree_r[2])
        if branching_cost + branching_alpha < leaf_cost + alpha:
            return ({ list(tree.keys())[0]: [min_cost_subtree_l[0], min_cost_subtree_r[0]] }, obs_index, branching_cost, branching_alpha)
        return ((self._assign_leaf_val(obs_index), obs_index), obs_index, leaf_cost, alpha)

    def _prune(self, alpha=0):
        if len(self.tree_dict) == 0:
            self.fit()
        return self._get_min_cost_subtree(self.tree_dict, alpha)[0]

    def prune(self, alphas=np.arange(0, 2.2, 0.2), k=10):
        """
        Prunes a fitted decision tree
        :param alphas: (numpy.ndarray) Alpha values for cost complexity pruning
        :param k: (int) Number of folds to use in cross-validation for selecting alpha
        """
        if self.pruned and ((alphas.shape == self.pruning_alphas.shape) or (np.not_equal(alphas, self.pruning_alphas).sum() == 0)):
            return
        self.pruning_alphas = alphas
        x, y = self.data[self.features], self.data.dependent
        best_subtrees = pd.Series(alphas, index=alphas).apply(lambda alpha: self._prune(alpha))
        self.fold_mean_errors = pd.DataFrame(index=range(1, k+1), columns=alphas)
        for i in range(1, k+1):
            exclude, include = cv.partition_indices(i, k, self.data.index)
            tree_i = DecisionTree(x.loc[include], y.loc[include], max_features=self.max_features, max_depth=self.max_depth, min_samples=self.min_samples)
            tree_i.fit()
            best_subtrees_i = pd.Series(alphas, index=alphas).apply(lambda alpha: tree_i._prune(alpha))
            self.fold_mean_errors.loc[i] = best_subtrees_i.apply(lambda tree: utils.get_mean_error(tree_i.predict(x.loc[exclude,:], tree),
              y.loc[exclude], not(self.is_classifier)))
        mean_FMEs = self.fold_mean_errors.mean().values
        self.tree_dict = best_subtrees[alphas[np.where(mean_FMEs == mean_FMEs.min())[-1][-1]]]
        self.pruned = True

