import matplotlib.pyplot as plt
from data_preprocessing import data_preprocess
import seaborn as sns

def data_visualization():

    data = data_preprocess()
    # data['default'].value_counts().plot.bar(figsize=(10,5),title='Classes Split for Dataset')
    # plt.xlabel('Classes')
    # plt.ylabel('Count')
    #col=list(data.columns)
    # for i in col:
    #     #plt.figure(figsize=(14,12))
    #     sns.boxplot(y=i,data=data)
    # plt.show()
    # for i in col:
    #     #plt.figure(figsize=(14,12))
    #     sns.distplot(x=data[i])
    #     plt.title(i)
    # plt.show()
    sns.heatmap(data.corr(), linewidth=1, annot=True,  cmap="coolwarm", fmt=".4f")
    plt.show()
    return data

data_visualization()