import pandas as pd
import numpy as np 


def remove_rows_with_missing_ratings(df):
    df = df.dropna(subset=['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating'])
    return df

def combine_description_strings(df):
    df["Description"] = df["Description"].str.replace("'About this space', ", "")
    df["Description"] = df["Description"].str.replace("''", "")
    return df

def set_default_feature_values(df):
    """
    Replace Null value with "1".

    Parameters:
        (pd.DataFrame)

    Returns:
        (pd.DataFrame)
    """
    df["guests"].fillna(1, inplace=True)
    df["beds"].fillna(1, inplace=True)
    df["bathrooms"].fillna(1, inplace=True)
    df["bedrooms"].fillna(1, inplace=True)
    return df

def clean_tabular_data(raw_dataset):
    cleaned_dataset  = remove_rows_with_missing_ratings(raw_dataset)
    cleaned_dataset  = combine_description_strings(cleaned_dataset)
    cleaned_dataset  = set_default_feature_values(cleaned_dataset)
    return(cleaned_dataset)

def load_airbnb(data_file, label_column):
    """
    Load Airbnb data and return features and labels.

    Parameters:
        data_file (str): Path to the CSV data file.
        label_column (str): Name of the label column.

    Returns:
        (pd.DataFrame, pd.Series): Tuple containing features (DataFrame) and labels (Series).
    """
    # Load the data into a DataFrame
    data = pd.read_csv(data_file)
    
    # Separate the label column from the features
    features = data.drop(columns=[label_column])
    labels = data[label_column]
    
    return features, labels


if __name__ == "__main__":
    raw_data = pd.read_csv("listing.csv")
    cleaned_data = clean_tabular_data(raw_data)
    cleaned_data.to_csv('clean_tabular_data.csv', index=False)
    cleaned_data.to_string('clean_tabular_data.txt', index=False)
    

