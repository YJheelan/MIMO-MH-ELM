# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 16:22:44 2025

@author: yjheelan
"""
import numpy as np
import pandas as pd
import time
from numpy.linalg import pinv

# Global configuration
CONFIG = {
    'data_file': "Data.csv",
    'window_size': 48,
    'num_hidden': 1000,
    'num_initializations': 1,
    'max_horizon': 24,
    'train_ratio': 0.8,
    'lambda_reg': 1e-6,  # Ridge regularization
    'output_names': ['Total_MW', 'Thermal_MW', 'Hydro_MW', 'Micro_Hydro_MW', 
                     'Solar_MW', 'Wind_MW', 'BioEner_MW', 'Import_MW'],
    'energy_cols': [2, 3, 5, 6, 7, 8, 9, 11],
    'reconciliation': True,  # Enabling WLS reconciliation
}

def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses energy time-series data for SISO, MIMO, MIMO-MH, MIMO-MH-WLS.
        
        This function performs comprehensive data preparation including temporal feature engineering,
        data cleaning, and matrix construction suitable for energy forecasting models.
        
        Arguments:
            file_path (str): Path to the CSV file containing energy time-series data.
                            Expected to have a 'Date' column and energy consumption columns.
        
        Process:
            1. Data Loading:
               - Reads CSV file and cleans column names (strips whitespace)
               - Converts 'Date' column to UTC datetime format
               - Sets datetime as DataFrame index for time-series operations
            
            2. Data Cleaning:
               - Removes original 'Date' column to avoid duplication
               - Fills missing values with zeros to maintain data continuity
            
            3. Temporal Feature Engineering:
               - Extracts hour information (1-24 format) from datetime index
               - Applies cyclic transformation using sine/cosine encoding:
                 * Hours_sin = sin(2π × hour / 24)
                 * Hours_cos = cos(2π × hour / 24)
               - This encoding captures the cyclical nature of time
            
            4. Energy Data Processing:
               - Selects energy columns based on CONFIG['energy_cols'] indices
               - Applies maximum(value, 0) to ensure non-negative energy values
               - Stacks selected columns into a unified energy data matrix
            
            5. Feature Matrix Construction:
               - Combines energy data with cyclical time features
               - Creates final input matrix: [energy_features, Hours_sin, Hours_cos]
        
        Returns:
            tuple: A 3-element tuple containing:
                - input_matrix (np.ndarray): Feature matrix with shape (n_samples, n_features)
                  where features include energy variables and cyclical time components
                - num_rows (int): Total number of data points (samples) in the dataset
                - num_outputs (int): Number of target variables for prediction 
                  (derived from CONFIG['output_names'])
        
        Usage:
            Primary function for data preprocessing pipeline in energy forecasting systems.
            Prepares raw time-series data for training/testing machine learning models
            that predict energy consumption patterns.
            
            Example:
                X, n_samples, n_targets = load_and_preprocess_data('energy_data.csv')
        
        Dependencies:
            - pandas (pd): For DataFrame operations and datetime handling
            - numpy (np): For numerical operations and matrix construction
            - CONFIG: Global configuration dictionary containing 'energy_cols' and 'output_names'
        
        Note:
            The cyclical encoding of time features is crucial for models to understand
            temporal relationships in energy consumption patterns (daily/seasonal cycles).
    """
    print("Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    # Date processing
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df.index = df["Date"]
    df.drop(['Date'], axis=1, inplace=True)
    df = df.fillna(0)
    # Extracting hours and cyclic transformation
    Hours = df.index.hour + 1  # de 1 à 24
    
    Hours_sin = np.sin(2 * np.pi * Hours / 24)
    Hours_cos = np.cos(2 * np.pi * Hours / 24)
    
    # Energy data extraction
    energy_data = np.column_stack([
        np.maximum(df.iloc[:, i].values, 0) for i in CONFIG['energy_cols']
    ])
    
    # Creation of the input matrix
    input_matrix = np.column_stack([energy_data, Hours_sin, Hours_cos])
    
    num_rows = input_matrix.shape[0]
    num_outputs = len(CONFIG['output_names'])
    print(f"Data loaded: {num_rows} rows, {input_matrix.shape[1]} columns")
    
    return input_matrix, num_rows, num_outputs