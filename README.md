
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

### Clean a dataset
#### tabular_data.py: 
1. remove_rows_with_missing_ratings:
2. combine_description_strings: combines the list items into the same string. It take in the dataset as a pandas dataframe and return the same type. It removes any records with a missing description, and also remove the "About this space" prefix
3. set_default_feature_values: replace empty values in "guests", "beds", "bathrooms", and "bedrooms" columns with "1"
4. clean_tabular_data: takes in the raw dataframe, calls 3 functions sequentially on the output of the previous one, and returns the processed data.
5. load_airbnb: Load Airbnb data and return features and labels

### Regression model
#### modelling_regression.py:
1. import_and_standarised_data:Imports the data through the load_airbnb() function and then standardises it

2. split_data: Splits the data into training, validating and testing data
3. train_linear_regression_model: Trains a linear regression model and evaluates its performance.

4. custom_tune_regression_model_hyperparameters: Perform a grid search over hyperparameter values for a regression model.
5. tune_regression_model_hyperparameters: Tune the hyperparameters of the Random Forest Regression model using grid search.
6. save_model: Save the best regression model, its hyperparameters, and performance metrics to files.
7. evaluate_all_models: Evaluate multiple regression models with different hyperparameter combinations.
8. find_best_model: Find the best regression model among saved models.


### Classification model
#### modelling_classification.py:
1. Initialise
This project involves the implementation and evaluation of classification models for predicting categories of Airbnb property listings. The code is organized into a class, `ClassificationModel`, which encapsulates various methods for data processing, model evaluation, hyperparameter tuning, and model saving. 

2. Model Evaluation and Saving
Choose a model, a folder name for saving results, and then evaluate the model.

3. Finding the Best Model
To find the best classification model among a list of models, use the find_best_model function.

### Methods
1. import_and_standarise_data(data_file): Load Airbnb data. Handle missing values. Standardize numeric columns.
2. splited_data(X, y): Split data into training, validation, and test sets.
3. classification_metrics_performance(y_test, y_pred): Evaluate classification metrics (accuracy, precision, recall, F1 score).
4. confusion_matrix(model, X_test, y_test, y_pred): Generate and display the confusion matrix for the model.
5. tune_hyperparameters(model, X_train, X_val, X_test, y_train, y_val, y_test, hyperparameters) Perform hyperparameter tuning.
6. save_classification_model(folder_name, result_dict): Save the best classification model, hyperparameters, and metrics.
7. evaluate_model(model, hyperparameters_dict, folder_name): Evaluate a given classification model and save the results.

### Hyperparameters

- Logistic Regression: LogisticRegression_param
- Decision Tree Classifier: DecisionTreeCLassifier_param
- Gradient Boosting Classifier: GardientBoostingClassifer_pram
- Random Forest Classifier: RandomForestClassifier_pram

#### main.py:

#### db_creds.yaml & my_db_creds.yaml:



## Libraries


## Furthur work
This project involves the implementation and evaluation of classification models for predicting categories of Airbnb property listings. The code is organized into a class, `ClassificationModel`, which encapsulates various methods for data processing, model evaluation, hyperparameter tuning, and model saving.
