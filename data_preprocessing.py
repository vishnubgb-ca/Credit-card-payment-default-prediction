import pandas as pd
from data_analysis import data_analysis

def data_preprocess():
    data = data_analysis()
    data.drop("ID", axis=1, inplace =True)
    print(data)
    return data

data_preprocess()