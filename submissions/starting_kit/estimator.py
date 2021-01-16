import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, Ridge, Lasso
from sklearn.preprocessing import FunctionTransformer

def get_estimator():

    def get_cat(df):
        cat_cols = list()
        for col in set(df.columns):
            if col.endswith("is_ffpe"):
                ...
            elif df[col].dtype == object:
                cat_cols.append(col)
        return df[cat_cols]

    def get_bool(df):
        bool_cols = list()
        for col in set(df.columns):
            if col.endswith("is_ffpe"):
                bool_cols.append(col)
        return df[bool_cols]

    def get_num(df):
        num_cols = list()
        for col in set(df.columns):
            if df[col].dtype == np.float64:
                num_cols.append(col)
        return df[num_cols]


    cat_pipeline = make_pipeline(
        FunctionTransformer(get_cat), 
        SimpleImputer(strategy='constant', fill_value='missing'),
        OneHotEncoder(handle_unknown='ignore'),
    )
    bool_pipeline = make_pipeline(
        FunctionTransformer(get_bool), 
        SimpleImputer(strategy='most_frequent', fill_value='missing'),
        OrdinalEncoder(),
    )
    num_pipeline = make_pipeline(
        FunctionTransformer(get_num), 
        SimpleImputer(strategy='mean')
    )

    preprocessing = make_union(
        cat_pipeline,
        num_pipeline,
        bool_pipeline,
    )
    
    model = make_pipeline(
        preprocessing,
        RandomForestRegressor(n_estimators=100),
    )

    return model


class Regressor(BaseEstimator):
    
    def __init__(this):
        this.model = get_estimator()

    def fit(this, X, y):
        this.model.fit(X, y)
        return this

    def predict(this, X):
        return this.model.predict(X)
        
