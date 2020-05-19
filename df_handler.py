import pandas as pd
from pandas import DataFrame as df
from sklearn import preprocessing
import diabetes_corr as dc
import matplotlib.pyplot as plt


def read_csv(path):
    # Read the file
    df = pd.read_csv(path)
    df.dropna() # No NaN values
    return df


def negative_positive(df): # inspired by titanic 
    # convert class tested_negative or tested_positive to 0 or 1
    label_enc = preprocessing.LabelEncoder()
    df['Class'] = label_enc.fit_transform(df['Class'].astype(str))
    return df


def zero_values(df):
    print("Total number of rows and columns before cleaning = {0} ".format(df.shape))
    print("Number of rows missing - Pregnancies = {0}".format(len(df.loc[df['Pregnancies'] == 0])))
    print("Number of rows missing - Glucose = {0}".format(len(df.loc[df['Glucose'] == 0])))
    print("Number of rows missing - Bloodpressure  = {0}".format(len(df.loc[df['Bloodpressure'] == 0])))
    print("Number of rows missing - Skinthickness = {0}".format(len(df.loc[df['Skinthickness'] == 0])))
    print("Number of rows missing - Insulin = {0}".format(len(df.loc[df['Insulin'] == 0])))
    print("Number of rows missing - Body mass  = {0}".format(len(df.loc[df['Bodymass'] == 0])))
    print("Number of rows missing - Diabetes_pedigree_function = {0}".format(len(df.loc[df['Diabetes_pedigree_function'] == 0])))
    print("Number of rows missing - Age  = {0}\n".format(len(df.loc[df['Age'] == 0])))


    # https://stackoverflow.com/questions/40299055/pandas-how-to-fill-null-values-with-mean-of-a-groupby
def mean_val_insulin(df):
    df.value = df.groupby('Class')['Insulin'].apply(lambda x: x.fillna(x.mean()))
    df.value = df.value.fillna(df.value.mean())
    print(df.shape)


def new_dataframe(df):
    # Because 227 patients did had 0 skinthickness, we thought it would be best to remove this columns from the df,
    # Neither did it correlate much with other columns in the df
    del df['Skinthickness']

    new_df = df[
    (df['Pregnancies'] > 0) & 
    (df['Glucose'] > 0) & 
    (df['Bloodpressure'] > 0) &
    #(df['Insulin'] > 0) & 
    (df['Bodymass'] > 0) & 
    (df['Diabetes_pedigree_function'] > 0) & 
    (df['Age'] > 0)
    ]
    print("Total number of rows and columns after cleaning = {0} ".format(new_df.shape))
    dc.plot_corr(new_df) 
    new_df.corr()   
    # Fewer data but no zero values in the df. 
    # Note fx. that glycose correlates 49% with the class(diabetes)
    
    new_df.to_csv('newfile.csv', index=False)