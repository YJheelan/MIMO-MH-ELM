
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 18:30:32 2025

@author: yjheelan
"""

from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

from config import CONFIG
from training import get_metrics
def create_sequences_siso(energy_data: np.ndarray, variable_idx: int, 
                          window_size: int, prediction_horizon: int):
    """
    Creates input/output sequences for a SISO (Single Input Single Output) model.

    Args:
        energy_data (np.ndarray): Complete energy data matrix.
        variable_idx (int): Index of the target variable.
        window_size (int): Length of the input sequence window.
        prediction_horizon (int): Forecast horizon (in steps).

    Returns:
        tuple: Tuple (X_siso, Y_siso) where:
               - X_siso is the input sequence matrix.
               - Y_siso is the corresponding target vector.
    """
    num_rows = energy_data.shape[0]
    num_observations = num_rows - window_size - prediction_horizon
    print(f"{num_observations} observation")
    
    X_siso = np.zeros((num_observations, window_size))
    Y_siso = np.zeros(num_observations)
    
    for i in range(num_observations):
        X_siso[i, :] = energy_data[i:i+window_size, variable_idx]
        Y_siso[i] = energy_data[i+window_size+prediction_horizon-1, variable_idx]
    
    return X_siso, Y_siso

def train_elm_siso(X_train: np.ndarray, Y_train: np.ndarray, 
                   num_hidden: int, num_initializations: int):
    """
    Trains a SISO ELM (Extreme Learning Machine) model.

    Args:
        X_train (np.ndarray): Training input data.
        Y_train (np.ndarray): Training target values.
        num_hidden (int): Number of hidden neurons.
        num_initializations (int): Number of random initializations to try.

    Returns:
        tuple: Best (input_weights, bias, output_weights) based on RMSE.
    """
    best_rmse = float('inf')
    best_input_weights = None
    best_bias = None
    best_output_weights = None
    
    for init in range(num_initializations):
        input_weights = np.random.rand(num_hidden, X_train.shape[1])
        bias = np.random.rand(num_hidden, 1)
        
        H = np.maximum(0, X_train @ input_weights.T + bias.T)
        
        lambda_reg = 1e-6
        output_weights = np.linalg.solve(
            H.T @ H + lambda_reg * np.eye(H.shape[1]),
            H.T @ Y_train
        )
        
        Y_pred = np.maximum(H @ output_weights, 0)
        rmse = np.sqrt(np.mean((Y_pred - Y_train)**2))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_input_weights = input_weights
            best_bias = bias
            best_output_weights = output_weights
    
    return best_input_weights, best_bias, best_output_weights

def predict_elm_siso(X_test: np.ndarray, input_weights: np.ndarray, 
                     bias: np.ndarray, output_weights: np.ndarray):
    """
    Makes predictions using a trained SISO ELM model.

    Args:
        X_test (np.ndarray): Input test data.
        input_weights (np.ndarray): Trained input weights.
        bias (np.ndarray): Trained bias.
        output_weights (np.ndarray): Trained output weights.

    Returns:
        np.ndarray: Predicted values.
    """
    H_test = np.maximum(0, X_test @ input_weights.T + bias.T)
    Y_pred = np.maximum(H_test @ output_weights, 0)
    return Y_pred

def run_siso_experiments(input_matrix, num_rows, num_outputs):
    """
    Runs SISO experiments for all horizons and variables.

    Args:
        input_matrix (np.array): Input data matrix.
        num_rows (int): Total number of rows.
        num_outputs (int): Number of output variables (targets).

    Returns:
        list: A list of results, each entry in the format:
              [model_name, horizon, output_name, NRMSE, NMAE, NMBE, R²]
    """
    results = []
    output_names = CONFIG['output_names']
    energy_data = input_matrix[:, :num_outputs]
    
    for horizon in range(1, CONFIG['max_horizon'] + 1):
        print(f"\n--- Forecast horizon : {horizon}h ---")
        
        for j in range(num_outputs):
            variable_name = CONFIG['output_names'][j]
            print(f"  > Training for the variable : {variable_name}")
            
            # Create sequences
            X_siso, Y_siso = create_sequences_siso(energy_data, j, CONFIG['window_size'], horizon)
            
            # Check that there is enough data
            if X_siso.shape[0] < 10:
                print(f"    Not enough data for the horizon {horizon}")
                continue
            
            # Train/test division
            train_size = int(CONFIG['train_ratio'] * X_siso.shape[0])
            X_train = X_siso[:train_size]
            Y_train = Y_siso[:train_size]
            X_test = X_siso[train_size:]
            Y_test = Y_siso[train_size:]
            
            # Training
            input_weights, bias, output_weights = train_elm_siso(
                X_train, Y_train, CONFIG['num_hidden'], CONFIG['num_initializations']
            )
            
            # Prediction
            Y_pred = predict_elm_siso(X_test, input_weights, bias, output_weights)
            
            # Metrics calculation
            nrmse, nmae, nmbe, r2 = get_metrics(Y_test, Y_pred)
            
            # Convert results to scalars if necessary
            if isinstance(nrmse, np.ndarray):
                nrmse = nrmse.item()
            if isinstance(nmae, np.ndarray):
                nmae = nmae.item()
            if isinstance(nmbe, np.ndarray):
                nmbe = nmbe.item()
            if isinstance(r2, np.ndarray):
                r2 = r2.item()
            
            results.append([
                'SISO', horizon, variable_name, 
                float(nrmse), float(nmae), float(nmbe), float(r2)
            ])
    
    return results