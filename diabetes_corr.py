import seaborn as sns


def plot_corr(df): 
    sns.heatmap(df.corr(), annot = True,fmt='.2f')