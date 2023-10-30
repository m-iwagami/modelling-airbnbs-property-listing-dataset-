import itertools
import json
import math
import os
import pandas as pd
import joblib
import numpy as np
from joblib import load
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
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
    """
    Trains a linear regression model and evaluates its performance.

    Parameters:
        X_train, X_validation, X_test (pandas.core.frame.DataFrame): DataFrames containing the features for training, validation, and testing.
        y_train, y_validation, y_test (pandas.core.series.Series): Series containing the targets/labels.
    """
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
    validation_R2 = []


    for hyperparameter_values in itertools.product(*hyperparameters_dict.values()):
        hyperparameters = dict(zip(hyperparameters_dict.keys(), hyperparameter_values))
        regression_model = model.set_params(**hyperparameters)
        model = regression_model.fit(X_train, y_train)
        y_pred_val = model.predict(X_validation)
        rmse_val = math.sqrt(mean_squared_error(y_validation, y_pred_val))
        validation_R2.append(metrics.r2_score(y_validation, y_pred_val))


        if rmse_val < best_rmse:
            best_model = model
            best_hyperparameters = hyperparameters
            best_rmse = rmse_val

    # Calculate RMSE on the test set for the best model
    y_pred_test = best_model.predict(X_test)
    rmse_test = math.sqrt(mean_squared_error(y_test, y_pred_test))
    test_R2 = metrics.r2_score(y_test, y_pred_test)


    # Create a dictionary of performance metrics
    performance_metrics = {
        "validation_RMSE": rmse_test,
        'R^2' : test_R2
    }

    return best_model, best_hyperparameters, performance_metrics

    



def tune_regression_model_hyperparameters(X,y):
    """
    Tune the hyperparameters of the Random Forest Regression model using grid search.

    Parameters:
        X (pandas.core.frame.DataFrame): Features of the dataset.
        y (pandas.core.series.Series): Target labels of the dataset.

    This function performs a grid search over various hyperparameter combinations for the Random Forest Regression model
    to find the best set of hyperparameters.

    The hyperparameter grid consists of:
    - 'n_estimators': Number of trees in the forest.
    - 'max_depth': Maximum depth of the trees.
    - 'min_samples_split': Minimum number of samples required to split an internal node.
    - 'min_samples_leaf': Minimum number of samples required to be at a leaf node.

    The best hyperparameters and the corresponding negative mean squared error score are printed.

    Note: The 'cv' parameter in GridSearchCV determines the number of folds for cross-validation.

    Returns:
        None
    """
    random_forest_model = RandomForestRegressor()
    parameter_grid =  {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(
        estimator = random_forest_model,
        param_grid = parameter_grid, 
        cv=5,
        scoring='neg_mean_squared_error'
    )
    grid_search.fit(X.values, y)
    print(f"Training and tuning RandomForestRegressor...")
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Score (Negative Mean Squared Error):", grid_search.best_score_)

def save_model(folder_name, best_model, best_hyperparameters, performance_metric):
    """
    Save the best regression model, its hyperparameters, and performance metrics to files.

    Parameters:
        folder_name (str): The name of the folder to save the model.
        best_model: The best-trained regression model.
        best_hyperparameters (dict): A dictionary of the best hyperparameter values.
        performance_metric (dict): A dictionary of performance metrics.

    This function saves the best-trained regression model to a .joblib file, the best hyperparameters to a JSON file,
    and the performance metrics to another JSON file in the specified folder.

    Returns:
        None
    """
    models_dir = 'modelling-airbnbs-property-listing-dataset-/models'

    # Create Models folder
    current_dir = os.path.dirname(os.getcwd())
    models_path = os.path.join(current_dir, models_dir)
    if os.path.exists(models_path) == False:
        os.mkdir(models_path)

    # Create regression folder
    regression_dir = 'modelling-airbnbs-property-listing-dataset-/models/regression'
    current_dir = os.path.dirname(os.getcwd())
    regression_path = os.path.join(current_dir, regression_dir)
    if os.path.exists(regression_path) == False:
        os.mkdir(regression_path)

    # Create linear_regression folder
    folder_name_dir = os.path.join(regression_path,folder_name)
    current_dir = os.path.dirname(os.getcwd())
    folder_name_path = os.path.join(current_dir, folder_name_dir)
    if os.path.exists(folder_name_path) == False:
        os.mkdir(folder_name_path)


    # Save the model to a .joblib file
    joblib.dump(best_model, os.path.join(folder_name_path,"model.joblib"))

    #Save the hyperparameters to a JSON file
    hyperparameters_filename = os.path.join(folder_name_path, "hyperparameters.json")
    with open (hyperparameters_filename, 'w') as json_file:
        json.dump(best_hyperparameters, json_file)
    
    #Save the metrics to a JSON file
    metrics_filename = os.path.join(folder_name_path, "metrics.json")
    with open (metrics_filename, 'w') as json_file:
        json.dump(performance_metric, json_file)

    return

def evaluate_all_models(models, hyperparamaters_dict):
    """
    Evaluate multiple regression models with different hyperparameter combinations.

    Parameters:
        models (list): List of regression models to evaluate.
        hyperparameters_dict (list of dict): List of dictionaries containing hyperparameter grids for each model.

    This function evaluates each regression model in the 'models' list with a range of hyperparameter combinations
    specified in 'hyperparameters_dict'. It performs grid search for each model and stores the best model, its
    hyperparameters, and performance metrics in separate folders.

    Returns:
        None
    """
    

    # Import and standarise data
    data_file = "clean_tabular_data.csv"
    X, y = import_and_standarised_data(data_file)
    # Split data
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(X, y)

    for i in range(len(models)):
        best_regression_model, best_hyperparameters_dict, performance_metrics = custom_tune_regression_model_hyperparameters(models[i], X_train, X_validation, X_test, y_train, y_validation, y_test, hyperparameters_dict[i])
        
        # Print Results
        print(best_regression_model, best_hyperparameters_dict, performance_metrics)
        
        #Save the best model
        folder_name= str(models[i])[0:-2]
       

        save_model(folder_name, best_regression_model, best_hyperparameters_dict, performance_metrics)

    return
 
def find_best_model(models):
    """
    Find the best regression model among saved models.

    Parameters:
        models (list): List of regression models to evaluate.

    This function searches for saved regression models in the specified folders, loads them, and compares their performance
    metrics. It returns the best regression model, its hyperparameters, and the corresponding metrics.

    Returns:
        best_regression_model: The best-trained regression model.
        best_hyperparameters_dict (dict): A dictionary of the best hyperparameter values.
        best_metrics_dict (dict): A dictionary of performance metrics.
    """
    best_regression_model = None
    best_hyperparameters_dict = {}
    best_metrics_dict = {}
    
    regression_dir = "modelling-airbnbs-property-listing-dataset-/models/regression"
    current_dir = os.path.dirname(os.getcwd())
    regression_path = os.path.join(current_dir, regression_dir)

    for i in range(len(models)):
        model_str = str(models[i])[0:-2]
        model_dir = os.path.join(regression_path, model_str)
        model = load(os.path.join(model_dir, 'model.joblib'))
        with open (os.path.join(model_dir, 'hyperparameters.json'),'r') as hyperparameters_path:
            hyperparameters = json.load(hyperparameters_path)
        
        with open(os.path.join(model_dir, 'metrics.json'), 'r') as metrics_path:
            metrics = json.load(metrics_path)

        if best_regression_model is None or metrics.get("R^2") > best_metrics_dict.get("R^2"):
            best_regression_model = model
            best_hyperparameters_dict = hyperparameters
            best_metrics_dict = metrics

    return best_regression_model, best_hyperparameters_dict, best_metrics_dict


models = [SGDRegressor(), DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor()]

hyperparameters_dict = [{ #SGDRegressor Hyperparameters (Selection)

    'loss':['squared_error','huber', 'squared_epsilon_insensitive'],
    'penalty':['l2', 'l1', 'elasticnet'],
    'alpha':[0.0001, 0.001],
    'l1_ratio':[0.15, 0.2],
    'fit_intercept':[True, False],
    'max_iter' :[1000],
    'shuffle' :[True, False],
    'early_stopping':[True, False]

},
                        { # DecisionTreeRegressor Hyperparameters (Selection)
    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter':['best', 'random'],
    'max_features':[10]

},
                        { # RandomForestRegressor Hyperparameters (Selection)
    'n_estimators':[50, 100,],
    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'bootstrap':[True, False],
    'max_features':[10]
},

                        { # GradientBoostingRegressor Hyperparameters (Selection)
    'loss':['squared_error','huber'],
    'learning_rate':[0.1, 0.2, 0.5],
    'n_estimators':[50, 100, 200,],
    'criterion':['squared_error', 'friedman_mse'],
    'max_features':[10],

}]

if __name__ == "__main__":

    evaluate_all_models(models, hyperparameters_dict)
    best_regression_model, best_hyperparameters_dict, best_metrics_dict = find_best_model(models)
    
    print("Best Regression Model:")
    print(best_regression_model)
    print("Hyperparameters:")
    print(best_hyperparameters_dict)
    print("Metrics:")
    print(best_metrics_dict)

