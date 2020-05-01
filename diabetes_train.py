from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB


def train_data(diabetes_data):
    feature_col_names = ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']            
    predicted_class_names = ['class']

    X = diabetes_data[feature_col_names].values 
    y = diabetes_data[predicted_class_names].values 

    split_test_size = 0.3

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = split_test_size,random_state=42)
    return X_train, X_test, y_train,y_test


def impute_with_mean(X_train, X_test):
    fill_0 = SimpleImputer(missing_values=0,strategy="mean")

    X_train= fill_0.fit_transform(X_train)
    X_test = fill_0.fit_transform(X_test)
    return X_train, X_test

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


