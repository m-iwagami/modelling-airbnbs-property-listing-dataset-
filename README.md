
#  Modelling Airbnb's property listing dataset
    - A Data Science project from AiCore course

## Project Introduction
Build a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team

## Language and Tools
- python
- github
- AWS

## Project Outline
1. Clean a dataset
2. Create a regression model
3. Create a classification model
4. Create a configurable neural network


## Requirements
- pip install requirements.txt
- python3 

## Modules

### 1. Clean a dataset
#### tabular_data.py: 
1. remove_rows_with_missing_ratings:
2. combine_description_strings: combines the list items into the same string. It take in the dataset as a pandas dataframe and return the same type. It removes any records with a missing description, and also remove the "About this space" prefix
3. set_default_feature_values: replace empty values in "guests", "beds", "bathrooms", and "bedrooms" columns with "1"
4. clean_tabular_data: takes in the raw dataframe, calls 3 functions sequentially on the output of the previous one, and returns the processed data.
5. load_airbnb: Load Airbnb data and return features and labels

### 2. Create a regression model
#### modelling,py:
1. import_and_standarised_data:Imports the data through the load_airbnb() function and then standardises it
'\n'
2. split_data: Splits the data into training, validating and testing data
3. train_linear_regression_model: Trains a linear regression model and evaluates its performance.

4. custom_tune_regression_model_hyperparameters: Perform a grid search over hyperparameter values for a regression model.
5. tune_regression_model_hyperparameters: Tune the hyperparameters of the Random Forest Regression model using grid search.
6. save_model: Save the best regression model, its hyperparameters, and performance metrics to files.
7. evaluate_all_models: Evaluate multiple regression models with different hyperparameter combinations.
8. find_best_model: Find the best regression model among saved models.


#### 

#### main.py:

#### db_creds.yaml & my_db_creds.yaml:



## Libraries


## Furthur work
