import pandas as pd
import numpy as np

def data_cleaning_tool(data):
    # Handling missing values
    data = handle_missing_values(data)

    # Removing duplicates
    data = remove_duplicates(data)

    # Standardizing data formats
    data = standardize_data_formats(data)

    # Perform additional data cleaning tasks
  

    return data

def handle_missing_values(data):
    # Drop rows with missing values
    data = data.dropna()

    # Fill missing values with appropriate techniques
    # data = data.fillna(value) or data = data.interpolate()

    return data

def remove_duplicates(data):
    # Drop duplicate rows
    data = data.drop_duplicates()

    return data

def standardize_data_formats(data):
    # Convert 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Remove unwanted symbols or characters from specific columns
    data['Amount'] = data['Amount'].str.replace('$', '')

    # Perform additional data format standardization
   

    return data

# Example usage
# Load the data from a CSV file
data = pd.read_csv('data.csv')

# Clean the data using the Data Cleaning Tool
cleaned_data = data_cleaning_tool(data)

# Display the cleaned data
print(cleaned_data.head())
