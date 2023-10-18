from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from data_preprocessing import data_preprocess

def feature_engineering():
    le=LabelEncoder()
    data = data_preprocess()
    # print(data)
    X=data.drop(columns="default.payment.next.month")
    y=data['default.payment.next.month']
    target = le.fit_transform(np.ravel(y))
    sm = SMOTE()
    X_upd, y_upd = sm.fit_resample(X, target.ravel())
    data_new=X_upd
    data_new['default.payment.next.month']=y_upd 
    data_new.to_csv("default_prediction.csv",index=False)
    return data

feature_engineering()