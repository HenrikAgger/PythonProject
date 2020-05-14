from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import diabetes_corr as dc
from pandas import DataFrame as df
import matplotlib.pyplot as plt


def train_data(df):
    feature_col_names = ['Pregnancies','Glucose','Bloodpressure','Skinthickness','Insulin','Bodymass','Diabetes_pedigree_function','age']            
    predicted_class_names = ['class']

    X = df[feature_col_names].values 
    y = df[predicted_class_names].values 

    split_test_size = 0.3

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = split_test_size,random_state=10)
    return X_train, X_test, y_train,y_test


def naive_bayes(X_train, y_train):
    nb_model = GaussianNB()

    nb_model.fit(X_train, y_train.ravel())
    return nb_model


def models_acuracy(X_train,nb_model):
    nb_predict_train = nb_model.predict(X_train)
    return nb_predict_train
    

def models_acuracy_X_Test(X_test, nb_model):
    nb_predict_test=nb_model.predict(X_test)
    return nb_predict_test

def zero_values(df):
    print("Total number of rows = {0}\n".format(len(df)))
    print("Number of rows missing - Pregnancies = {0}".format(len(df.loc[df['Pregnancies'] == 0])))
    print("Number of rows missing - Glucose = {0}".format(len(df.loc[df['Glucose'] == 0])))
    print("Number of rows missing - Bloodpressure  = {0}".format(len(df.loc[df['Bloodpressure'] == 0])))
    print("Number of rows missing - Skinthickness = {0}".format(len(df.loc[df['Skinthickness'] == 0])))
    print("Number of rows missing - Insulin = {0}".format(len(df.loc[df['Insulin'] == 0])))
    print("Number of rows missing - Body mass  = {0}".format(len(df.loc[df['Bodymass'] == 0])))
    print("Number of rows missing - Diabetes_pedigree_function = {0}".format(len(df.loc[df['Diabetes_pedigree_function'] == 0])))
    print("Number of rows missing - Age  = {0}\n".format(len(df.loc[df['age'] == 0])))


def new_dataframe(df):
    new_df = df[
    (df['Pregnancies'] > 0) & 
    (df['Glucose'] > 0) & 
    (df['Bloodpressure'] > 0) & 
    (df['Skinthickness'] > 0) & 
    (df['Insulin'] > 0) & 
    (df['Bodymass'] > 0) & 
    (df['Diabetes_pedigree_function'] > 0) & 
    (df['age'] > 0)
    ]
    print("Total number of rows after cleaning = {0}".format(len(new_df)))
    dc.plot_corr(new_df) 
    new_df.corr()   # Fewer data but no zero values in the df. 
                # Note fx. that glycose correlates 51,6% with the class(diabetes)
    print(new_df)
