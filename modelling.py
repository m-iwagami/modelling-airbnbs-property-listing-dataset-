
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

def custom_tune_regression_model_hyperparameters(model_class, X_train, X_validation, X_test, y_train, y_validation, y_test, hyperparameters_dict):
  '''
   return the best model, a dictionary of its best hyperparameter values, and a dictionary of its performance metrics
   
   '''
  validation_RMSE = {}
  validation_R2 = {}
  model_hyperparameters_val = {}
  model_val = {}

  for i in range(len(model_class)): 
    model = model_class[i]
    hyperparameters_dict = hyperparameters_dict[i]

    for hyperparameter_value in itertools.product(*hyperparameters_dict.value()):
      hyperparameters = dict(zip(hyperparameters_dict.keys(), hyperparameter_value))
      regression_model = model_class(**hyperparameters)
      model =  regression_model.fit(X_train, y_train)
      y_pred_val = model.predict(X_validation)
      validation_RMSE.append(math.sqrt(mean_squared_error(y_validation, y_pred_val), squared=False))
      validation_R2.append(r2_score(y_validation, y_pred_val))
      model_hyperparameters_val.append(hyperparameters)
      model_val.append(regression_model)
      
      # Update best model if the current model has a lower RMSE
      if validation_RMSE < best_rmse:
            best_model = model
            best_hyperparameters = hyperparameters
            best_rmse = validation_RMSE
    
    # Calculate RMSE on the test set for the best model
    y_pred_test = best_model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    best_regression_mode = best_model
    
    # Create a dictionary of performance metrics
    performance_metrics = {
        "validation_RMSE": best_rmse,
        "test_RMSE": rmse_test,
    }
    
    return best_model, best_hyperparameters, best_regression_mode, performance_metrics
if __name__=="__main__":
    data_file = "clean_tabular_data.csv"
    train_linear_regression_model(data_file)
    custom_tune_regression_model_hyperparameters()


