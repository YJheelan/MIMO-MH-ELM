# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 16:23:25 2025

@author: yjheelan
"""

"""
Module for data processing and creation of sliding windows
"""
import numpy as np
from config import CONFIG

def sliding_window_mimo(input_matrix, pred_horizon, window_size, num_rows ):
    """
    Creates sliding windows for single-horizon prediction
    
    Args:
        X (np.array): Input matrix
        pred_horizon (int): Prediction horizon
        window_size (int): Window size
        num_rows (int): Number of rows in the dataset
    Returns:
        tuple: (X (windowed), Y (target))
    """
    
    obs = num_rows - window_size - pred_horizon
    X = np.zeros((obs, window_size * input_matrix.shape[1])) # Input with windows of window_size points
    Y = np.zeros((obs, input_matrix.shape[1] - 2))  # Output without sine/cosine
    for i in range(obs):
        window_data = input_matrix[i:i+window_size, :].T
        X[i, :] = window_data.flatten()  # flatten window into a single row
        # what we want is the energy production prediction_horizon steps in the future
        Y[i, :] = input_matrix[i+window_size+pred_horizon-1, :-2]   # All columns except sin/cos
    return np.nan_to_num(X), np.nan_to_num(Y)

def split_train_test(X, Y, train_ratio=0.8):
    """
    Splits data into training and testing sets
    
    Args:
        X (np.array): Input data
        Y (np.array): Output data
        train_ratio (float): Proportion for training
        
    Returns:
        tuple: (X_train, Y_train, X_test, Y_test)
    """
    trainSize = round(0.8 * X.shape[0])
    X_train = X[:trainSize, :]
    Y_train = Y[:trainSize, :]
    X_test = X[trainSize:, :]
    Y_test = Y[trainSize:, :]
    print("split_train_test :")
    print("train size", trainSize)
    print("len(X_train)", len(X_train))
    print("len(Y_train)", len(Y_train))
    print("len(X_test)",len(X_test))
    print(" len(Y_test)", len(Y_test))

    print(f".shape X_train",X_train.shape)
    print(f".shape Y_train",Y_train.shape)
    print(f".shape X_test",X_test.shape)
    print(f".shape Y_test",Y_test.shape)
    return X_train, Y_train, X_test, Y_test

def sliding_window_mimo_mh(X, max_horizon, window_size, num_rows):
    """
    Creates sliding windows for MIMO_MH prediction
    
    Args:
        X (np.array): Input matrix
        max_horizon (int): Maximum prediction horizon
        window_size (int): Window size
        num_rows (int): Number of rows in the dataset
        
    Returns:
        tuple: (X_windowed, Y_mimo)
    """
    print("sliding_window_mimo_mh :") 
    obs = num_rows - window_size - max_horizon
    print("observation window",obs)
    X_windowed = np.lib.stride_tricks.sliding_window_view(X, (window_size, X.shape[1]))[:obs]
    X_windowed = X_windowed.reshape(obs, -1)
    print("len(X_windowed)",len(X_windowed))

    Y_mimo = np.stack([
        X[window_size + h : window_size + h + obs, :-2]
        for h in range(max_horizon)
    ], axis=1)
    print("len(Y_mimo) : multi-input (sliding window)",len(Y_mimo))

    return np.nan_to_num(X_windowed), np.nan_to_num(Y_mimo)

def split_train_test(X, Y, train_ratio=0.8):
    """
    Splits data into training and testing sets
    
    Args:
        X (np.array): Input data
        Y (np.array): Output data
        train_ratio (float): Proportion for training
        
    Returns:
        tuple: (X_train, Y_train, X_test, Y_test)
    """
    train_size = int(train_ratio * X.shape[0])
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]
    print("split_train_test :")
    print("train size", train_size)
    print("len(X_train)", len(X_train))
    print("len(Y_train)", len(Y_train))
    print("len(X_test)",len(X_test))
    print(" len(Y_test)", len(Y_test))

    return X_train, Y_train, X_test, Y_test
