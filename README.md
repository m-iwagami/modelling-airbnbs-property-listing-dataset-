###

## Description of the project:

- Build a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team
1.  Clean the data from 




tabular_data.py: 
remove_rows_with_missing_ratings()


 what it does, the aim of the project, and what you learned
Installation instructions
Usage instructions
File structure of the project
License information


#  Modelling Airbnb's property listing dataset
    - A Data Science project from AiCore course
## Project Introduction
Build a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team

## Project Outline
1. Clean a dataset
2. 

## Requirements
- pip install requirements.txt
- python3 
## Modules

#### tabular_data.py: 
1. remove_rows_with_missing_ratings:
2. combine_description_strings: combines the list items into the same string. It take in the dataset as a pandas dataframe and return the same type. It removes any records with a missing description, and also remove the "About this space" prefix
3. set_default_feature_values: replace empty values in "guests", "beds", "bathrooms", and "bedrooms" columns with "1"
4. clean_tabular_data: takes in the raw dataframe, calls 3 functions sequentially on the output of the previous one, and returns the processed data.
5. load_airbnb: Load Airbnb data and return features and labels


#### 
#### 

#### main.py:

#### db_creds.yaml & my_db_creds.yaml:



## Libraries


## Furthur work
