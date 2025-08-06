# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 17:36:59 2025

@author: yjheelan
"""

import numpy as np
from config import CONFIG
from training import get_metrics
def run_persistence_models(input_matrix, num_rows, num_outputs):
    """
    Computes evaluation metrics for two persistence baseline models:
    - 'Persistence': uses the value from h steps before as the prediction at horizon h.
    - 'Persistence-24h': always uses the value from 24h ago regardless of horizon.

    These baselines are used to benchmark the performance of forecasting models.

    Args:
        input_matrix (np.array): The full input data matrix containing observations (targets and other columns).
        num_rows (int): Total number of rows in the input matrix.
        num_outputs (int): Number of output variables (targets) to forecast.

    Returns:
        list: A list of results, each entry in the format:
              [model_name, horizon, output_name, NRMSE, NMAE, NMBE, R²]
    """
    results = []
    output_names = CONFIG['output_names']
    window_size = CONFIG['window_size']
    max_horizon = CONFIG['max_horizon']
    train_ratio = CONFIG['train_ratio']
    print(f"\n{'='*60}")
    print("Processing persistence horizon and persistence 24)...")

    for horizon in range(1, max_horizon + 1):
        print(f" → Persistance horizon {horizon}h")
        # Extract Y only (no need for X for persistence)
        num_obs = num_rows - window_size - horizon
        Y = np.zeros((num_obs, input_matrix.shape[1] - 2))
        for i in range(num_obs):
            Y[i, :] = input_matrix[i + window_size + horizon - 1, :-2]
        Y = np.nan_to_num(Y)

        # Split train/test
        train_size = int(train_ratio * Y.shape[0])
        Y_test = Y[train_size:, :]

        # Generate persistence predictions
        Y_pers = np.zeros_like(Y_test)
        Y_pers_24h = np.zeros_like(Y_test)
        for i in range(Y_test.shape[0]):
            if i - horizon >= 0:
                Y_pers[i] = Y_test[i - horizon]
            # sinon reste à 0
            if i - 24 >= 0:
                Y_pers_24h[i] = Y_test[i - 24]

        # Compute metrics for each variable
        for j in range(num_outputs):
            y_true = Y_test[:, j]
            y_hor = Y_pers[:, j]
            y_24h = Y_pers_24h[:, j]

            nrmse1, nmae1, nmbe1, r21 = get_metrics(y_true, y_hor)
            nrmse2, nmae2, nmbe2, r22 = get_metrics(y_true, y_24h)

            results.append(['Persistance_h', horizon, output_names[j],
                            float(nrmse1), float(nmae1), float(nmbe1), float(r21)])
            results.append(['Persistance_24h', horizon, output_names[j],
                            float(nrmse2), float(nmae2), float(nmbe2), float(r22)])

    return results