
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import itertools
from tabular_data import load_airbnb


def train_linear_regression_model(data_file):
    # Load the Airbnb data with "Price_Night" as the label
    label_column = "Price_Night"

    features, labels = load_airbnb(data_file, label_column)

    numerical_features = features.select_dtypes(include=['number'])

    np.random.seed(10)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
    numerical_features, labels, test_size=0.2, random_state=42
    )
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)


    # Create a pipeline for feature scaling and the SGDRegressor
    eta0 = 1e-2
    sgd_pipeline = Pipeline([("feature_scaling", StandardScaler()),
                             ("sgd", SGDRegressor(random_state=42, eta0 = eta0))])
    sgd_pipeline.fit(X_train, y_train)
    y_pred = sgd_pipeline.predict(X_test)
    y_pred_train = sgd_pipeline.predict(X_train) 

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
  '''
   return the best model, a dictionary of its best hyperparameter values, and a dictionary of its performance metrics
   
   '''
  validation_RMSE = {}
  validation_R2 = {}
  model_hyperparameter_val = {}
  model_val = {}


  for hyperparameter_value in itertools.product(*hyperparameters_dict.value()):
    hyperparameter = dict(zip(hyperparameters_dict.keys(), hyperparameter_value))
    model.set_params(**hyperparameter)
    model.fit(X, y)
    score = model.score(X, y)
    if score > best_score:
          best_score = score
          best_params = hyperparameter
    return best_params, best_score



if __name__=="__main__":
    data_file = "clean_tabular_data.csv"
    train_linear_regression_model(data_file)



