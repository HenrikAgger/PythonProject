import pandas as pd
from pandas import DataFrame as df
from sklearn import preprocessing
import matplotlib.pyplot as plt


def plot_corr(df,size=10): 
    # Yellow means that they are highly correlated
    corr = df.corr() 
    fig, ax = plt.subplots(figsize=(size,size))
    ax.matshow(corr) 
    plt.xticks(range(len(corr.columns)),corr.columns) 
    plt.yticks(range(len(corr.columns)),corr.columns)