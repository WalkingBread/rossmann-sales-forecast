import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

def rmspe(actual, predicted):
    return rmse(actual, predicted) / np.mean(actual)

def r2(actual, predicted):
    return r2_score(actual, predicted)