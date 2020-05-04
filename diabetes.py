import pandas as pd
from pandas import DataFrame as df
from sklearn import preprocessing
import matplotlib.pyplot as plt


def read_csv(path):
    # Read the file
    df = pd.read_csv(path)
    df.dropna() # No NAN values
    return df


def negative_positive(df):
    # convert class tested_negative or tested_positive to 0 or 1
    label_enc = preprocessing.LabelEncoder()
    df['class'] = label_enc.fit_transform(df['class'].astype(str))
    return df


def plot_corr(df,size=10): 
    # Yellow means that they are highly correlated
    corr = df.corr() 
    fig, ax = plt.subplots(figsize=(size,size))
    ax.matshow(corr) 
    plt.xticks(range(len(corr.columns)),corr.columns) 
    plt.yticks(range(len(corr.columns)),corr.columns)