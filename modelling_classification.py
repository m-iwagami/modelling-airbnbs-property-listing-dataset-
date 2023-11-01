import numpy as np
import math
import pandas as pd
import sklearn 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from modelling_regression import split_data
from sklearn.impute import SimpleImputer



from tabular_data import load_airbnb
from tabular_data import remove_rows_with_missing_ratings

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


def logic_regression(X,y):
    
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(X,y)
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)
    predictions = logisticRegr.predict(X_test)
    score = logisticRegr.score(X_test, y_test)
    print(f"Score: {score}"
          f"Pridictions: {predictions}")

   
    

if __name__ == "__main__":
    df = "listing.csv"
    X, y = import_and_standarise_data(df)
    logic_regression(X, y)
