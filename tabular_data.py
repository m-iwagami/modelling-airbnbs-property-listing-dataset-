import pandas as pd
import numpy as np 


def remove_rows_with_missing_ratings(df):
    """
    Remove rows with missing values in specific rating columns.

    Parameters:
        df (pd.DataFrame): The DataFrame containing Airbnb data.

    Returns:
        pd.DataFrame: A DataFrame with rows containing missing values in rating columns removed.
    """
    df = df.dropna(subset=['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating'])
    return df
    
def combine_description_strings(df):
    """
    Combine and clean the 'Description' column in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing Airbnb data.

    Returns:
        pd.DataFrame: A DataFrame with cleaned 'Description' column.
    """
    df["Description"] = df["Description"].str.replace("'About this space', ", "")
    df["Description"] = df["Description"].str.replace("''", "")
    df.dropna(subset=['Description'], inplace=True)
    return df

def set_default_feature_values(df):
    """
    Set default values for missing features in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing Airbnb data.

    Returns:
        pd.DataFrame: A DataFrame with missing feature values replaced with defaults.
    """
    df["guests"].fillna(1, inplace=True)
    df["beds"].fillna(1, inplace=True)
    df["bathrooms"].fillna(1, inplace=True)
    df["bedrooms"].fillna(1, inplace=True)
    
    # Remove rows where 'guests' column equals "Somerford Keynes England United Kingdom"
    df["guests"] = pd.to_numeric(df["guests"], errors='coerce').fillna(0).astype(int)
    df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors='coerce').fillna(0).astype(int)

    return df


def clean_tabular_data(raw_dataset):
    """
    Clean and preprocess tabular Airbnb data.

    Parameters:
        raw_dataset (pd.DataFrame): The raw DataFrame containing Airbnb data.

    Returns:
        pd.DataFrame: A cleaned and processed DataFrame.
    """
    cleaned_dataset = remove_rows_with_missing_ratings(raw_dataset)
    cleaned_dataset = combine_description_strings(cleaned_dataset)
    cleaned_dataset = set_default_feature_values(cleaned_dataset)
    cleaned_dataset = cleaned_dataset.drop(columns = ['Unnamed: 19','url','ID','Category', 'Title','Location', 'Amenities','Description'])
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
    cleaned_dataset = clean_tabular_data(raw_data)
    cleaned_dataset.to_csv('clean_tabular_data.csv', index=False)
    
