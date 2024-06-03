# data_preparation.py

"""
Data Preparation Module for Secure Federated Learning Systems

This module contains functions for collecting, cleaning, normalizing, and preparing data for training federated learning models.

Techniques Used:
- Data cleaning
- Normalization
- Feature extraction

Libraries/Tools:
- pandas
- numpy
- scikit-learn

"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreparation:
    def __init__(self, raw_data_dir='data/raw/', processed_data_dir='data/processed/', test_size=0.2):
        """
        Initialize the DataPreparation class.
        
        :param raw_data_dir: str, directory containing raw data
        :param processed_data_dir: str, directory to save processed data
        :param test_size: float, proportion of the dataset to include in the test split
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.test_size = test_size

    def load_data(self, filename):
        """
        Load data from a CSV file.
        
        :param filename: str, name of the CSV file
        :return: DataFrame, loaded data
        """
        filepath = os.path.join(self.raw_data_dir, filename)
        data = pd.read_csv(filepath)
        return data

    def clean_data(self, data):
        """
        Clean the data by removing null values and duplicates.
        
        :param data: DataFrame, input data
        :return: DataFrame, cleaned data
        """
        data = data.dropna().drop_duplicates()
        return data

    def normalize_data(self, data):
        """
        Normalize the data using standard scaling.
        
        :param data: DataFrame, input data
        :return: DataFrame, normalized data
        """
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data)
        return pd.DataFrame(normalized_data, columns=data.columns)

    def split_data(self, data, target_column):
        """
        Split the data into training and test sets.
        
        :param data: DataFrame, input data
        :param target_column: str, name of the target column
        :return: tuple, training and test sets (X_train, X_test, y_train, y_test)
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def save_data(self, data, filename):
        """
        Save the processed data to a CSV file.
        
        :param data: DataFrame, processed data
        :param filename: str, name of the output CSV file
        """
        os.makedirs(self.processed_data_dir, exist_ok=True)
        filepath = os.path.join(self.processed_data_dir, filename)
        data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

    def preprocess(self, filename, target_column):
        """
        Execute the full preprocessing pipeline.
        
        :param filename: str, name of the raw data file
        :param target_column: str, name of the target column
        """
        # Load data
        data = self.load_data(filename)
        
        # Clean data
        data = self.clean_data(data)
        
        # Normalize data
        data = self.normalize_data(data)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(data, target_column)
        
        # Save processed data
        self.save_data(pd.concat([X_train, y_train], axis=1), 'train_data.csv')
        self.save_data(pd.concat([X_test, y_test], axis=1), 'test_data.csv')
        print("Data preprocessing completed.")

if __name__ == "__main__":
    preprocessing = DataPreparation(raw_data_dir='data/raw/', processed_data_dir='data/processed/', test_size=0.2)
    
    # Execute the preprocessing pipeline
    preprocessing.preprocess('raw_data.csv', target_column='target')
    print("Data preprocessing completed and data saved.")
