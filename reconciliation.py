# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 10:39:33 2025

@author: yjheelan
"""
import numpy as np
from config import CONFIG
from numpy.linalg import pinv

def create_aggregation_matrix():
    """
    Creates the aggregation matrix S representing the energy hierarchy.

    Hierarchical structure:
        Total = Thermal + Hydro + Micro_Hydro + Solar + Wind + BioEner + Import

    Returns:
        np.array: Aggregation matrix of shape (8 x 7), where:
                  - The first row corresponds to the total (sum of all 7 sources).
                  - Rows 1 to 7 are identity rows for each individual source.
    """
    num_sources = 7  # Disaggregated sources (excluding Total)
    num_total = 8    # Including Total
    
    # Matrice S: (8 x 7) - Total en haut, puis les sources
    S = np.zeros((num_total, num_sources))
    
    # Row 0: Total = sum of all sources
    S[0, :] = 1  # Total = sum of all sources
    # Rows 1-7: each source = itself
    S[1:, :] = np.eye(num_sources)
    
    return S

def estimate_error_variances(Y_true, Y_pred):
    """
    Estimates the forecast error variances for each variable.

    Used to build the weighting matrix W⁻¹ for WLS reconciliation.

    Args:
        Y_true (np.array): Ground truth values (shape: n_samples x n_variables)
        Y_pred (np.array): Model predictions (shape: n_samples x n_variables)

    Returns:
        np.array: Vector of error variances (1D, length = n_variables),
                  with a minimum value of 1e-6 to avoid division by zero.
    """
    errors = Y_true - Y_pred
    # Calculation of variances by variable and time horizon
    variances = np.var(errors, axis=0)
    # Avoid zero variances
    variances = np.maximum(variances, 1e-6)
    return variances

def wls_reconciliation(y_forecasts, S, W_inv):
    """
    Applies Weighted Least Squares (WLS) reconciliation to forecasts 
    to ensure consistency with the defined hierarchical structure.

    Reconciliation equation:
    b_WLS = (S^T W^-1 S)^-1 S^T W^-1 y_MH
    ỹ = S b_WLS

    Args:
        y_forecasts (np.array): Initial unreconciled forecasts (shape: n_samples x n_variables)
        S (np.array): Aggregation matrix defining the hierarchy (shape: n_variables x n_bottom)
        W_inv (np.array): Inverse of the error covariance matrix (shape: n_variables x n_variables)

    Returns:
        np.array: Reconciled forecasts with the same shape as y_forecasts.
    """
    n_samples = y_forecasts.shape[0]
    reconciled_forecasts = np.zeros_like(y_forecasts)
    
    for i in range(n_samples):
        y_mh = y_forecasts[i]
        
        # WLS: b_WLS = (S^T W^-1 S)^-1 S^T W^-1 y_MH
        try:
            StW_invS = S.T @ W_inv @ S
            StW_inv_y = S.T @ W_inv @ y_mh
            
            # Linear system resolution
            b_wls = np.linalg.solve(StW_invS, StW_inv_y)
            
            # Reconciled forecasts: y_tilde = S * b_WLS
            reconciled_forecasts[i] = S @ b_wls
            
        except np.linalg.LinAlgError:
            # Fallback with pseudo-inverse if matrix is singular
            try:
                StW_invS_pinv = pinv(S.T @ W_inv @ S)
                b_wls = StW_invS_pinv @ S.T @ W_inv @ y_mh
                reconciled_forecasts[i] = S @ b_wls
            except:
                # As a last resort, keep the original forecasts.
                reconciled_forecasts[i] = y_mh
    
    # Ensuring positive energy forecasts
    reconciled_forecasts = np.maximum(reconciled_forecasts, 0)
    
    return reconciled_forecasts