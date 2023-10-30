import numpy as np
import math
import pandas as pd
import sklearn 
from sklearn.preprocessing import StandardScaler
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
    X = remove_rows_with_missing_ratings(X)
  # Select only numeric columns
    numeric_columns = X.select_dtypes(include=[np.number])

  # Standadize numeric column
    #std = StandardScaler()
    #scaled_features = std.fit_transform(numeric_columns.values)
    #X[numeric_columns.columns] = scaled_features
    
    print(X,y)
    return X,y


if __name__ == "__main__":
    df = "listing.csv"
    X, y = import_and_standarise_data(df)
