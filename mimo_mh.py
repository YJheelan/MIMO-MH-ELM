# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 14:45:54 2025

@author: yjheelan
"""
import numpy as np
from config import CONFIG
from data_processing import sliding_window_mimo_mh, split_train_test
from training import train_elm, predict_elm, get_metrics
from reconciliation import create_aggregation_matrix, estimate_error_variances, wls_reconciliation

# Version without reconciliation
def run_mimo_multihorizon_simple(input_matrix, num_rows, num_outputs):
    """
    Runs a basic (without reconciliation) multi-horizon MIMO experiment.

    Args:
        input_matrix (np.array): Input matrix
        num_rows (int): Number of rows
        num_outputs (int): Number of output variables

    Returns:
        list: MIMO experiment results, test values, and predictions
    """
    results = []
    output_names = CONFIG['output_names']
    max_horizon = CONFIG['max_horizon']
    
    print(f"\n{'='*60}")
    print("MIMO MULTI-HORIZON EXPERIMENT")
    
    # Data preparation
    X_mh, Y_mh = sliding_window_mimo_mh(
        input_matrix, max_horizon, CONFIG['window_size'], num_rows
    )
    X_train_mh, Y_train_mh, X_test_mh, Y_test_mh = split_train_test(
        X_mh, Y_mh, CONFIG['train_ratio']
    )
    
    # Reshape target for training
    Y_train_flat = Y_train_mh.reshape(Y_train_mh.shape[0], -1)
    Y_test_flat = Y_test_mh.reshape(Y_test_mh.shape[0], -1)
    
    # Training
    print("Training MIMO multi-horizon model...")
    model = train_elm(X_train_mh, Y_train_flat, X_test_mh, Y_test_flat)
    
    # Prediction
    Y_pred_flat = predict_elm(X_test_mh, model)
    Y_pred_mh = Y_pred_flat.reshape(Y_test_mh.shape)
    
    # Evaluation per horizon and variable
    for h in range(max_horizon):
        for i in range(num_outputs):
            nrmse, nmae, nmbe, r2 = get_metrics(Y_test_mh[:, h, i], Y_pred_mh[:, h, i])
            results.append([
                'MIMO-MH', h + 1, output_names[i], 
                float(nrmse), float(nmae), float(nmbe), float(r2)
            ])

    return results, Y_test_mh, Y_pred_mh

# Version with reconciliation
def run_mimo_multihorizon_experiment(input_matrix, num_rows, num_outputs):
    """
    Runs a MIMO multi-horizon experiment with optional WLS reconciliation.

    Args:
        input_matrix (np.array): Input matrix
        num_rows (int): Number of rows
        num_outputs (int): Number of output variables

    Returns:
        tuple: Original results, optionally reconciled results, true values, predictions, and reconciled predictions
    """
    results = []
    results_reconciled = []
    output_names = CONFIG['output_names']
    max_horizon = CONFIG['max_horizon']
    
    print("MIMO MULTI-HORIZON EXPERIMENT")
    # Data preparation
    X_mh, Y_mh = sliding_window_mimo_mh(
        input_matrix, max_horizon, CONFIG['window_size'], num_rows
    )
    X_train_mh, Y_train_mh, X_test_mh, Y_test_mh = split_train_test(
        X_mh, Y_mh, CONFIG['train_ratio']
    )
    # Reshape target for training
    Y_train_flat = Y_train_mh.reshape(Y_train_mh.shape[0], -1)
    Y_test_flat = Y_test_mh.reshape(Y_test_mh.shape[0], -1)
    
    print("Training MIMO multi-horizon model...")
    model = train_elm(X_train_mh, Y_train_flat, X_test_mh, Y_test_flat)
    # Prediction
    Y_pred_flat = predict_elm(X_test_mh, model)
    Y_pred_mh = Y_pred_flat.reshape(Y_test_mh.shape)
    
    # WLS Reconciliation if enabled
    if CONFIG['reconciliation']:
        print("Applying WLS reconciliation...")
        
        # Create aggregation matrix
        S = create_aggregation_matrix()
        
        # Initialize reconciled prediction array
        Y_pred_reconciled = np.zeros_like(Y_pred_mh)
        
        for h in range(max_horizon):
            # Estimate error variances for this horizon
            variances = estimate_error_variances(Y_test_mh[:, h, :], Y_pred_mh[:, h, :])
            W_inv = np.diag(1.0 / variances)
            
            # Apply WLS reconciliation
            Y_pred_reconciled[:, h, :] = wls_reconciliation(
                Y_pred_mh[:, h, :], S, W_inv
            )
    
    # Compute metrics for original predictions
    for h in range(max_horizon):
        for i in range(num_outputs):
            nrmse, nmae, nmbe, r2 = get_metrics(Y_test_mh[:, h, i], Y_pred_mh[:, h, i])
            results.append([
                'MIMO-MH', h + 1, output_names[i], 
                float(nrmse), float(nmae), float(nmbe), float(r2)
            ])
    
    # Calcul des métriques pour les prévisions réconciliées
    if CONFIG['reconciliation']:
        for h in range(max_horizon):
            for i in range(num_outputs):
                nrmse, nmae, nmbe, r2 = get_metrics(Y_test_mh[:, h, i], Y_pred_reconciled[:, h, i])
                results_reconciled.append([
                    'MIMO-MH-WLS', h + 1, output_names[i], 
                    float(nrmse), float(nmae), float(nmbe), float(r2)
                ])
        
        print("WLS reconciliation complete.")
        return results, results_reconciled, Y_test_mh, Y_pred_mh, Y_pred_reconciled
    
    return results, Y_test_mh, Y_pred_mh
