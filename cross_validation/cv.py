# Note: For classification only. As yet no support for regression CV.

import numpy as np
import pandas as pd
from sklearn.preprocessing import add_dummy_feature
from sklearn.utils import shuffle

# Function definitions to avoid recomputing moments for each data subset

def means_wo_obs(means, n, obs):
    return ((means * n - obs.sum()) / (n - obs.shape[0]))

def stderrs_wo_obs(means, stderrs, n, obs):
    return np.sqrt(n * stderrs**2 - ((obs - means)**2).sum()) / (n - obs.shape[0])

def restandardise_wo_obs(data, means, stderrs, n, obs):
    return ((data - means_wo_obs(means, n, obs)) / stderrs_wo_obs(means, stderrs, n, obs))

def get_shuffled_data(x, y):
    features = x.columns
    data = x
    data['dependent'] = y
    shuffled = shuffle(data).reset_index()
    data.drop(columns='dependent')
    return shuffled[features], shuffled.dependent

def partition_indices(i, k, index):
    if i < 1 or i > k:
        raise ValueError
    step, n = 1.0 / k, len(index)
    lb, ub = round((i-1) * step * n), round(i * step * n)
    include = list(range(lb))
    include.extend(range(ub, n))
    exclude = list(range(lb, ub))
    return index[exclude], index[include]

# Leave-one-out cross validation
# Example: loocv(x, y, LogisticRegression(multi_class='multinomial', solver='newton-cg'))
def loocv(x, y, model, shuffle=True, standardise=True, **kwargs):
    if shuffle:
        x, y = get_shuffled_data(x, y)
    n = x.shape[0] # Assumes x.shape[0] == y.shape[0]
    if standardise:
        x_means = x.mean()
        # Pandas gives adjusted stderr; need unadjusted to avoid taking sqrt of negative number
        x_stderrs = np.sqrt(x.std()**2 * (n-1)/n)
    predictions = np.full(n, -1)
    errors = np.zeros(n)
    for i in range(n):
        x_fit = x[x.index != i]
        if standardise:
            x_fit = restandardise_wo_obs(x[x.index != i], x_means, x_stderrs, n, x.loc[i])
        if intercept:
            x_fit = add_dummy_feature(x_fit)
        fitted = model.fit(x_fit, y[y.index != i], **kwargs)
        x_predict = np.reshape(x.loc[i].values, (1, -1))
        if intercept:
            x_predict = add_dummy_feature(x_predict)
        predictions[i] = fitted.predict(x_predict)[0]
        errors[i] = int(predictions[i] != y[i])
    cv_error = errors.mean()
    return (predictions, errors, cv_error)

def k_fold_cv(x, y, k, model, cv_verbose=False, shuffle=True, intercept=True, standardise=True, **kwargs):
    if shuffle:
        x, y = get_shuffled_data(x, y)
    n = x.shape[0] # Assumes x.shape[0] == y.shape[0]
    x_means = x.mean()
    x_stderrs = np.sqrt(x.std()**2 * (n-1)/n)
    predictions = np.full(n, -1)
    errors = np.zeros(n)
    fold_MSEs = np.zeros(k)
    for i in range(1, k+1):
        if cv_verbose:
            print('fold ', i)
        exclude, include = partition_indices(i, k, x.index)
        x_fit = x.loc[include]
        if standardise:
            x_fit = restandardise_wo_obs(x.loc[include], x_means, x_stderrs, n, x.loc[exclude])
        if intercept:
            x_fit = add_dummy_feature(x_fit)
        fitted = model.fit(x_fit, y.loc[include], **kwargs)
        exclude_len = len(x.loc[exclude])
        exclude_std = x.loc[exclude].values
        if standardise:
            exclude_std = ((x.loc[exclude] - x.loc[exclude].mean()) / np.sqrt(x.loc[exclude].std()**2 * (exclude_len-1)/exclude_len)).values
        if intercept:
            exclude_std = add_dummy_feature(exclude_std)
        predictions[exclude] = fitted.predict(exclude_std)
        errors[exclude] = (predictions[exclude] != y.values[exclude]).astype(int)
        fold_MSEs[i-1] = errors[exclude].mean()
    cv_error = fold_MSEs.mean()
    return (predictions, errors, cv_error)
