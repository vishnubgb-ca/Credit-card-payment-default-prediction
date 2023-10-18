import  pandas as pd

def loading_data():
    data = pd.read_csv('./UCI_Credit_Card.csv')
    return data

loading_data()