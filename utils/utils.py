import numpy as np

def is_numeric(var, int_as_numeric=True):
    """
    Determines whether variable is numeric
    :param var: Obesrvations of single variable
    :type var: Union[array, Series]
    :return: (bool) Is variable numeric?
    """
    return (int_as_numeric and (var.dtype.kind in np.typecodes["AllInteger"])) or (var.dtype.kind in np.typecodes["AllFloat"])

def get_mean_error(predicted, observed, numeric=None):
    if numeric is None:
        numeric = is_numeric(observed, False)
    if numeric:
        return np.square(predicted - observed).mean()
    return (predicted != observed).astype(int).mean()

