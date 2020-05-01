import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt


def read_csv(path):
    # Read the file
    diabetes_data = pd.read_csv(path)
    diabetes_data.dropna() # No NAN values
    return diabetes_data


def negative_positive(diabetes_data):
    # convert class tested_negative or tested_positive to 0 or 1
    label_enc = preprocessing.LabelEncoder()
    diabetes_data['class'] = label_enc.fit_transform(diabetes_data['class'].astype(str))
    return diabetes_data


def plot_corr(df,size=11): 
    # Yellow means that they are highly correlated
    corr = df.corr() 
    fig, ax = plt.subplots(figsize=(size,size))
    ax.matshow(corr) 
    plt.xticks(range(len(corr.columns)),corr.columns) 
    plt.yticks(range(len(corr.columns)),corr.columns) 



