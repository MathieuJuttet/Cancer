import pandas as pd
import numpy as np
import rampwf as rw
from sklearn.model_selection import ShuffleSplit
import os

problem_title = 'cancer_prediction'
_target_column_name = 'time'
_ignore_column_names = ["Unnamed: 0", "survivalEstimate", "case_id", 
                        "id_x", "id_y", 'diagnoses.0.days_to_last_follow_up']

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()

# An object implementing the workflow
workflow = rw.workflows.Estimator()

score_types = [rw.score_types.RMSE(name='rmse')]

def get_cv(X, y):
    cv = ShuffleSplit(n_splits=4, test_size=0.2, random_state=57)
    return cv.split(X, y)

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].to_numpy()
    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)
    return X_df, y_array

def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)

def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
