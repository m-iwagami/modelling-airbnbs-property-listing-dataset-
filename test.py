import itertools
import math
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import math
import itertools
from tabular_data import load_airbnb

def import_and_standarised_data(data_file):
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
   # Load the Airbnb data with "Price_Night" as the label
    label_column = "Price_Night"
    X, y = load_airbnb(data_file, label_column)
  # Select only numeric columns
    numeric_columns = X.select_dtypes(include=[np.number])

  # Standadize numeric column
    std = StandardScaler()
    scaled_features = std.fit_transform(numeric_columns.values)
    X[numeric_columns.columns] = scaled_features
    return X,y


def split_data(X, y):

    '''
        Splits the data into training, validating and testing data

        Parameters
        ----------
        X: pandas.core.frame.DataFrame
            A pandas DataFrame containing the features of the model

        y: pandas.core.series.Series
            A pandas series containing the targets/labels 

        Returns
        -------
        X_train, X_validation, X_test: pandas.core.frame.DataFrame
            A set of pandas DataFrames containing the features of the model

        y_train, y_validation, y_test: pandas.core.series.Series
            A set of pandas series containing the targets/labels  
    '''

    np.random.seed(10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

    print("Number of samples in:")
    print(f" - Training: {len(y_train)}")
    print(f" - Validation: {len(y_validation)}")
    print(f" - Testing: {len(y_test)}")
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test

def train_linear_regression_model(X_train, X_validation, X_test, y_train, y_validation, y_test):

    linear_regression_model_SDGRegr = SGDRegressor()
    model = linear_regression_model_SDGRegr.fit(X_train, y_train)

    y_pred = model.predict(X_test) 

    y_pred_train = model.predict(X_train) 

    # Evaluate the model's performance on the test set
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared (R2) Score: {r2}")

    mse = mean_squared_error(y_train, y_pred_train)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_train, y_pred_train)
    print("\n""Training Metrics")
    print(f"RMSE: {rmse}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared (R2) Score: {r2}")
    


def custom_tune_regression_model_hyperparameters(model, X_train, X_validation, X_test, y_train, y_validation, y_test, hyperparameters_dict):
    """
    Perform a grid search over hyperparameter values for a regression model.

    Parameters:
        model: The regression model.
        X_train, X_validation, X_test: Training, validation, and test feature sets.
        y_train, y_validation, y_test: Training, validation, and test target values.
        hyperparameters_dict: A dictionary of hyperparameter names mapping to a list of values to be tried.

    Returns:
        best_model: The best-trained model.
        best_hyperparameters: A dictionary of the best hyperparameter values.
        performance_metrics: A dictionary of performance metrics.
    """
    best_model = None
    best_hyperparameters = None
    best_rmse = float('inf')

    for hyperparameter_values in itertools.product(*hyperparameters_dict.values()):
        hyperparameters = dict(zip(hyperparameters_dict.keys(), hyperparameter_values))
        regression_model = model.set_params(**hyperparameters)
        model = regression_model.fit(X_train, y_train)
        y_pred_val = model.predict(X_validation)
        rmse_val = math.sqrt(mean_squared_error(y_validation, y_pred_val))

        if rmse_val < best_rmse:
            best_model = model
            best_hyperparameters = hyperparameters
            best_rmse = rmse_val

    # Calculate RMSE on the test set for the best model
    y_pred_test = best_model.predict(X_test)
    rmse_test = math.sqrt(mean_squared_error(y_test, y_pred_test))

    # Create a dictionary of performance metrics
    performance_metrics = {
        "validation_RMSE": best_rmse,
        "test_RMSE": rmse_test,
    }

    return best_model, best_hyperparameters, performance_metrics

def main():
    hyperparameters_dict = {
        'alpha': [0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.3, 0.5],
        'max_iter': [100, 200, 300],
    }

    data_file = "clean_tabular_data.csv"
    X, y = import_and_standarised_data(data_file)
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(X, y)
    
    # Create instances of model classes
    models = [SGDRegressor()]
#              , DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor()]

    for model in models:
        print(f"Training and tuning {model.__class__.__name__}...")
        best_model, best_hyperparameters, performance_metrics = custom_tune_regression_model_hyperparameters(model, X_train, X_validation, X_test, y_train, y_validation, y_test, hyperparameters_dict)
        
        # Print and/or store the best hyperparameters and performance metrics for each model.
        print(f"Best Hyperparameters for {model.__class__.__name__}: {best_hyperparameters}")
        print(f"Performance Metrics for {model.__class__.__name__}: {performance_metrics}")

if __name__ == "__main__":
    main()
