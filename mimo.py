# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 14:42:13 2025

@author: yjheelan
"""
from config import CONFIG
from data_processing import sliding_window_mimo, split_train_test
from training import train_elm, predict_elm, get_metrics

def run_mimo_single_horizon_experiments(input_matrix, num_rows, num_outputs):
    """
    Runs experiments for each prediction horizon individually.

    Args:
        input_matrix (np.array): Input matrix
        num_rows (int): Number of rows in the dataset
        num_outputs (int): Number of output variables

    Returns:
        list: Experiment results, true values, and predictions
    """
    results = []
    output_names = CONFIG['output_names']
    max_horizon = CONFIG['max_horizon']
    
    print(f"\n{'='*60}")
    print("MIMO EXPERIMENTS: SINGLE HORIZON")
    
    for horizon in range(1, max_horizon + 1):
        print(f"Processing horizon {horizon}h...")
        
        # Data preparation
        X_single, Y_single = sliding_window_mimo(
            input_matrix, horizon, CONFIG['window_size'], num_rows
        )
        X_train, Y_train, X_test, Y_test = split_train_test(
            X_single, Y_single, CONFIG['train_ratio']
        )
        
        # Training
        model = train_elm(X_train, Y_train, X_test, Y_test)
        
        # Prediction
        Y_pred = predict_elm(X_test, model)
        
        # Evaluation per variable
        for i in range(num_outputs):
            nrmse, nmae, nmbe, r2 = get_metrics(Y_test[:, i], Y_pred[:, i])
            results.append([
                'MIMO', horizon, output_names[i], 
                float(nrmse), float(nmae), float(nmbe), float(r2)
            ])
    
    return results, Y_test, Y_pred