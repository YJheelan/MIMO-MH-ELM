# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 16:23:50 2025

@author: yjheelan
"""

"""
ELM (Extreme Learning Machine) Model Training Module
"""
import numpy as np
from numpy.linalg import pinv
from config import CONFIG

def train_elm(X_train, Y_train, X_test, Y_test, num_hidden=None, num_initializations=None):
    """
    Trains an ELM model with multiple initializations

    Args:
        X_train (np.array): Training input data
        Y_train (np.array): Training output data
        X_test (np.array): Test input data
        Y_test (np.array): Test output data
        num_hidden (int): Number of hidden neurons
        num_initializations (int): Number of initializations

    Returns:
        tuple: (W, b, beta) - Parameters of the best model
    """
    if num_hidden is None:
        num_hidden = CONFIG['num_hidden']
    if num_initializations is None:
        num_initializations = CONFIG['num_initializations']
    
    best_rmse = float('inf')
    best_model = None
    best_inputWeights = None  # Initialise Best input weights
    best_bias = None  # Initialise Best bias
    best_outputWeights = None  # Initialise output weights
    
    for init in range(num_initializations):
        # Random weight initialization
        W = np.random.rand(num_hidden, X_train.shape[1]) #inputWeights
        b = np.random.rand(num_hidden, 1) #bias
        
        # Hidden layer computation (ReLU activation)
        H = np.maximum(0, X_train @ W.T + b.T)
        '''
        # Calcul des poids de sortie
        # Régularisation Ridge lambda = 1e-6
        HTH = H.T @ H
        HTY = H.T @ Y_train
        lambda_reg = CONFIG['lambda_reg']
        try:
            beta = np.linalg.solve(HTH + lambda_reg * np.eye(HTH.shape[0]), HTY) # outputWeights + ridge
        except np.linalg.LinAlgError:
            beta = pinv(H) @ Y_train  # outputWeights
        '''    
        beta = pinv(H) @ Y_train  # outputWeights
        # Test prediction
        H_test = np.maximum(0, X_test @ W.T + b.T)
        Y_pred = H_test @ beta # Unnormalized predictions
        Y_pred = np.maximum(Y_pred, 0)
        
        # RMSE computation
        rmse = np.mean(np.sqrt(np.mean(( Y_test - Y_pred)**2, axis=0)))
        
        # Save best model
        if rmse < best_rmse:
            best_model = (W, b, beta)
            best_rmse = rmse
            best_inputWeights = W  # Initialiser les meilleurs poids d'entrée
            best_bias = b  # Initialiser le meilleur biais
            best_outputWeights = beta
        #print("train_elm")
    return best_model


def predict_elm(X, model):
    """
    Predicts outputs using a trained ELM model

    Args:
        X (np.array): Input data
        model (tuple): (W, b, beta) - Trained model parameters

    Returns:
        np.array: Predictions
    """
    W, b, beta = model
    H = np.maximum(0, X @ W.T + b.T)
    Y_pred = np.maximum(H @ beta, 0)
    return Y_pred

def get_metrics(y_true, y_pred):
    """
    Computes evaluation metrics

    Args:
        y_true (np.array): Ground truth values
        y_pred (np.array): Predicted values

    Returns:
        tuple: (nrmse, nmae, nmbe, r2)
    """
    # Dimension management
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Calculating basic metrics
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    mbe = np.mean(y_true - y_pred, axis=0)
    mean_y = np.mean(y_true, axis=0)
    
    # Normalisation and other metrics
    with np.errstate(divide='ignore', invalid='ignore'):
        nrmse = np.nan_to_num(rmse / mean_y)
        nmae = np.nan_to_num(mae / mean_y)
        nmbe = np.nan_to_num(mbe / mean_y)
        
        # Coefficient of determination R²
        ss_res = np.sum((y_true - y_pred)**2, axis=0)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
        r2 = 1 - (ss_res / ss_tot)
        r2 = np.nan_to_num(r2)
    
    return nrmse, nmae, nmbe, r2