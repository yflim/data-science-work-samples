# Note: For classification only. As yet no support for regression CV.

import numpy as np
import pandas as pd

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
    return shuffled[features], shuffled.dependent

# Leave-one-out cross validation
# Example: loocv(LogisticRegression(multi_class='multinomial', solver='newton-cg'), x, y)
def loocv(x, y, model, shuffle=True):
    if shuffle:
        x, y = get_shuffled_data(x, y)
    n = x.shape[0] # Assumes x.shape[0] == y.shape[0]
    x_means = x.mean()
    # Pandas gives adjusted stderr; need unadjusted to avoid taking sqrt of negative number
    x_stderrs = np.sqrt(x.std()**2 * (n-1)/n)
    predictions = np.full(n, -1)
    errors = np.zeros(n)
    for i in range(n):
        fitted = model.fit(restandardise_wo_obs(x[x.index != i], x_means, x_stderrs, n, x.loc[i]), y[y.index != i])
        predictions[i] = fitted.predict(np.reshape(x.loc[i].values, (1, -1)))[0]
        errors[i] = int(predictions[i] != y[i])
    cv_error = errors.mean()
    return (predictions, errors, cv_error)

def k_fold_cv(x, y, k, model, shuffle=True):
    if shuffle:
        x, y = get_shuffled_data(x, y)
    n = x.shape[0] # Assumes x.shape[0] == y.shape[0]
    x_means = x.mean()
    x_stderrs = np.sqrt(x.std()**2 * (n-1)/n)
    i, step = 0, int(n * 1.0/k)
    predictions = np.full(n, -1)
    errors = np.zeros(n)
    fold_MSEs = np.zeros(step)
    for i in range(0, n, step):
        include = list(range(i))
        include.extend(range(i + 15, n))
        exclude = range(i, i + 15)
        fitted = model.fit(restandardise_wo_obs(x.loc[include], x_means, x_stderrs, n, x.loc[exclude]), y.loc[include])
        predictions[exclude] = fitted.predict(x.loc[exclude].values)
        errors[exclude] = (predictions[exclude] != y.values[exclude]).astype(int)
        fold_MSEs[int(i/step)] = errors[exclude].mean()
    cv_error = fold_MSEs.mean()
    return (predictions, errors, cv_error)
