import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from modelling_regression import split_data
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from tabular_data import load_airbnb

def import_and_standarise_data(data_file):
    '''
        Imports the data through the load_airbnb() function and then standardises it

        Parameters
        ----------
        data_file

        Returns
        -------
        X: pandas.core.frame.DataFrame
            A pandas DataFrame containing the features of the model

        y: pandas.core.series.Series
            A pandas series containing the targets/labels 
      '''
   # Load the Airbnb data with "Category" as the label
    label_column = "Category"
    X, y = load_airbnb(data_file, label_column)

    #X = remove_rows_with_missing_ratings(X)
  # Select only numeric columns
    numeric_columns = X.select_dtypes(include=[np.number])

  # Standadize numeric column
    std = StandardScaler()
    scaled_features = std.fit_transform(numeric_columns.values)
    X = pd.DataFrame(scaled_features, index=numeric_columns.index, columns=numeric_columns.columns)
  # Replace Null value with mean imputation
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    return X,y


def logistic_regression(X,y):
    
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(X,y)
    LogisticModel = LogisticRegression(random_state=0, max_iter=10000).fit(X_train, y_train)
    y_hat = LogisticModel.predict(X_test)
    predictions = LogisticModel.predict(X_test)
    score = LogisticModel.score(X_test, y_test)
    return y_test, y_hat

def classification_metrics_performance(y_test, y_hat):
    accuracy = accuracy_score(y_test, y_hat)
    precision = precision_score(y_test, y_hat, average="macro")
    recall = recall_score(y_test, y_hat, average="macro")
    f1 = f1_score(y_test, y_hat, average="macro")

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    
    train_metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }
    return train_metrics


if __name__ == "__main__":
    df = "listing.csv"
    X, y = import_and_standarise_data(df)
    y_test, y_hat = logistic_regression(X, y)
    train_metrics = classification_metrics_performance(y_test, y_hat)


