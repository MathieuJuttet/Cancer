import pandas as pd
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split

def split():
    df = pd.read_csv(pathlib.Path("./data/gdc_final_database.csv"))
    np.random.seed(0)
    df = df[-df["time"].isna()]
    data_train, data_test = train_test_split(df, train_size=1000)
    data_train.to_csv(pathlib.Path("./data/train.csv"))
    data_test.to_csv(pathlib.Path("./data/test.csv"))

split()
