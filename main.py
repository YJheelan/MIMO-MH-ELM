# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 16:24:42 2025

@author: yjheelan
"""

"""
Main Program - Energy Prediction with ELM
"""
import sys
# Set the path to the folder containing the main.py file.
# If needed, specify the path to the data.csv file in config.py
sys.path.append('Set the path to the folder containing the main.py file.')
import time
import pandas as pd

# Importing custom modules
from config import CONFIG
print("CONFIG loaded:", CONFIG['data_file'])
#Config
from config import load_and_preprocess_data
#Visualization
from visualization import (plot_model_comparison, plot_metrics_by_horizon, 
                           plot_heatmaps, print_best_results, 
                           plot_combined_mimo_comparison_advanced,
                           compare_models_performance)
# SISO Module SISO
from siso import run_siso_experiments
# MIMO Module 
from mimo import run_mimo_single_horizon_experiments
# MIMO-MH Module 
from mimo_mh import run_mimo_multihorizon_experiment, run_mimo_multihorizon_simple
# Persistence Module 
from persistence import run_persistence_models
#============================================================================
# Version that asks the user which models to choose

def display_menu():
    """Displays the experiment selection menu"""
    print("\n" + "="*60)
    print("EXPERIMENT SELECTION MENU")
    print("="*60)
    print("1 : Run all (SISO + MIMO + MIMO-MH + MIMO-MH-WLS)")
    print("2 : Run MIMO-MH + MIMO-MH-WLS")
    print("3 : Run MIMO + MIMO-MH-WLS")
    print("4 : Run SISO + MIMO + MIMO-MH-WLS")
    print("5 : Run only MIMO-MH-WLS")
    print("6 : Run only MIMO-MH")
    print("7 : Run only MIMO")
    print("8 : Run only SISO")
    print("9 : Custom model selection")
    print("="*60)

def get_user_choice():
    """Gets and validates the user's choice"""
    while True:
        try:
            choice = int(input("Please enter your choice (1-9): "))
            if 1 <= choice <= 9:
                return choice
            else:
                print("Invalid choice. Please enter a number between 1 and 9.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_custom_model_selection():
    """Allows custom selection of prediction models"""
    available_models = {
        1: "SISO",
        2: "MIMO", 
        3: "MIMO-MH",
        4: "MIMO-MH-WLS"
    }
    
    print("\n" + "="*50)
    print("CUSTOM MODEL SELECTION")
    print("="*50)
    
    # Ask how many models the user wants
    while True:
        try:
            num_models = int(input("How many models do you want to run (1-4)? "))
            if 1 <= num_models <= 4:
                break
            else:
                print("Please enter a number between 1 and 4.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    print(f"\nYou chose to run {num_models} model(s).")
    print("\nAvailable models:")
    for key, model in available_models.items():
        print(f"{key}: {model}")
    
    selected_models = []
    selected_indices = []
    
    for i in range(num_models):
        while True:
            try:
                print(f"\nSélection {i+1}/{num_models}")
                choice = int(input("Enter the model number: "))
                
                if choice not in available_models:
                    print("Invalid number. Please choose between 1 and 4.")
                    continue
                    
                if choice in selected_indices:
                    print(f"Model {available_models[choice]} has already been selected.")
                    print("Already selected models:", [available_models[idx] for idx in selected_indices])
                    continue
                
                selected_indices.append(choice)
                selected_models.append(available_models[choice])
                print(f"✓ {available_models[choice]} added to selection")
                break
                
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    print(f"\nSelected models: {', '.join(selected_models)}")
    return selected_models

def execute_selected_models(selected_models, input_matrix, num_rows, num_outputs):
    """
    Executes the models selected by the user.

    Args:
        selected_models (list): List of selected model names.
        input_matrix (np.ndarray): Preprocessed input data matrix.
        num_rows (int): Number of samples in the dataset.
        num_outputs (int): Number of output variables.

    Returns:
        tuple: Results and prediction matrices from the executed models.
    """
    print(f"\n{'='*60}")
    print(f"EXECUTING SELECTED MODELS: {', '.join(selected_models)}")
    print(f"{'='*60}")
    
    # Containers for results
    all_results = []
    siso_results = []
    mimo_results = []
    pers_results = []
    mimo_mh_results = []
    mimo_mh_wls_results = []
    
    # Variables for test data and predictions
    Y_test_mimo = None
    Y_pred_mimo = None
    Y_test_mh = None
    Y_pred_mh = None
    Y_pred_reconciled = None
    
    # Run models based on selection
    if "SISO" in selected_models:
        print(f"\n{'='*60}")
        print("PREDICTION WITH SISO")
        print(f"{'='*60}")
        siso_results = run_siso_experiments(input_matrix, num_rows, num_outputs)
        all_results.extend(siso_results)
    
    if "MIMO" in selected_models:
        print(f"\n{'='*60}")
        print("PREDICTION WITH MIMO")
        print(f"{'='*60}")
        mimo_results, Y_test_mimo, Y_pred_mimo = run_mimo_single_horizon_experiments(input_matrix, num_rows, num_outputs)
        all_results.extend(mimo_results)
    
    # Always include persistence as baseline
    if selected_models:  # At least one model selected
        pers_results = run_persistence_models(input_matrix, num_rows, num_outputs)
        all_results.extend(pers_results)
    
    if "MIMO-MH" in selected_models or "MIMO-MH-WLS" in selected_models:
        print(f"\n{'='*60}")
        print("PREDICTION WITH MIMO-MH" + (" and MIMO-MH-WLS" if "MIMO-MH-WLS" in selected_models else ""))
        print(f"{'='*60}")
        
        experiment_results = run_mimo_multihorizon_experiment(input_matrix, num_rows, num_outputs)
        
        if CONFIG['reconciliation'] and "MIMO-MH-WLS" in selected_models:
            mimo_mh_results, mimo_mh_wls_results, Y_test_mh, Y_pred_mh, Y_pred_reconciled = experiment_results
            
            if "MIMO-MH" in selected_models:
                all_results.extend(mimo_mh_results)
            if "MIMO-MH-WLS" in selected_models:
                all_results.extend(mimo_mh_wls_results)
                
        elif "MIMO-MH" in selected_models:
            if CONFIG['reconciliation']:
                mimo_mh_results, mimo_mh_wls_results, Y_test_mh, Y_pred_mh, Y_pred_reconciled = experiment_results
            else:
                mimo_mh_results, Y_test_mh, Y_pred_mh = experiment_results
            all_results.extend(mimo_mh_results)
        
        if "MIMO-MH-WLS" in selected_models and not CONFIG['reconciliation']:
            print("WARNING: WLS reconciliation is not enabled in the configuration.")
            print("MIMO-MH-WLS cannot be executed.")
    
    return all_results, Y_test_mimo, Y_pred_mimo, Y_test_mh, Y_pred_mh, Y_pred_reconciled, mimo_mh_results

def main():
    
    print("PREDICTION WITH SISO, MIMO, MIMO-MH AND WLS RECONCILIATION")
    
    # Display the menu and get user choice
    display_menu()
    user_choice = get_user_choice()
    
    print(f"\nSelected choice: {user_choice}")
    start_time = time.time()
    
    try:
        # 1. Load and preprocess data (always required)
        input_matrix, num_rows, num_outputs = load_and_preprocess_data(CONFIG['data_file'])
        
        # Variables to store results
        siso_results = []
        mimo_results = []
        pers_results = []
        mimo_mh_results = []
        mimo_mh_wls_results = []
        all_results = []
        
        # Variables for test data and predictions
        Y_test_mimo = None
        Y_pred_mimo = None
        Y_test_mh = None
        Y_pred_mh = None
        Y_pred_reconciled = None
        
        # New: Custom selection
        if user_choice == 9:
            selected_models = get_custom_model_selection()
            all_results, Y_test_mimo, Y_pred_mimo, Y_test_mh, Y_pred_mh, Y_pred_reconciled, mimo_mh_results = execute_selected_models(
                selected_models, input_matrix, num_rows, num_outputs
            )
        
        # Execution based on user choice (existing logic)
        elif user_choice == 1:  # Run everything
            print(f"\n{'='*60}")
            print("FULL EXECUTION - ALL MODELS")
            
            # SISO
            print(f"\n{'='*60}")
            print("PREDICTION WITH SISO")
            siso_results = run_siso_experiments(input_matrix, num_rows, num_outputs)
            
            # MIMO
            print(f"\n{'='*60}")
            print("PREDICTION WITH MIMO")
            mimo_results, Y_test_mimo, Y_pred_mimo = run_mimo_single_horizon_experiments(input_matrix, num_rows, num_outputs)
            
            # Persistence
            pers_results = run_persistence_models(input_matrix, num_rows, num_outputs)
            
            # MIMO-MH
            print(f"\n{'='*60}")
            print("PREDICTION WITH MIMO-MH")
            experiment_results = run_mimo_multihorizon_experiment(input_matrix, num_rows, num_outputs)
            if CONFIG['reconciliation']:
                mimo_mh_results, mimo_mh_wls_results, Y_test_mh, Y_pred_mh, Y_pred_reconciled = experiment_results
                all_results = mimo_mh_results + mimo_mh_wls_results + siso_results + pers_results + mimo_results
            else:
                mimo_mh_results, Y_test_mh, Y_pred_mh = experiment_results
                all_results = mimo_mh_results + siso_results + pers_results + mimo_results
        
        elif user_choice == 2:  # MIMO-MH + MIMO-MH-WLS
            print(f"\n{'='*60}")
            print("EXECUTION: MIMO-MH and MIMO-MH-WLS")
            
            experiment_results = run_mimo_multihorizon_experiment(input_matrix, num_rows, num_outputs)
            # Persistence
            pers_results = run_persistence_models(input_matrix, num_rows, num_outputs)
            if CONFIG['reconciliation']:
                mimo_mh_results, mimo_mh_wls_results, Y_test_mh, Y_pred_mh, Y_pred_reconciled = experiment_results
                all_results = mimo_mh_results + mimo_mh_wls_results + pers_results
            else:
                print("WARNING: WLS reconciliation is not enabled in configuration.")
                mimo_mh_results, Y_test_mh, Y_pred_mh = experiment_results
                all_results = mimo_mh_results + pers_results
        
        elif user_choice == 3:  # MIMO + MIMO-MH + MIMO-MH-WLS
            print(f"\n{'='*60}")
            print("EXECUTION: MIMO + MIMO-MH + MIMO-MH-WLS")
            
            # MIMO
            print(f"\n{'='*60}")
            print("PREDICTION WITH MIMO")
            mimo_results, Y_test_mimo, Y_pred_mimo = run_mimo_single_horizon_experiments(input_matrix, num_rows, num_outputs)
            
            # MIMO-MH
            print(f"\n{'='*60}")
            print("PREDICTION WITH MIMO-MH")
            experiment_results = run_mimo_multihorizon_experiment(input_matrix, num_rows, num_outputs)
            
            if CONFIG['reconciliation']:
                mimo_mh_results, mimo_mh_wls_results, Y_test_mh, Y_pred_mh, Y_pred_reconciled = experiment_results
                all_results =  mimo_mh_wls_results + mimo_results
            else:
                mimo_mh_results, Y_test_mh, Y_pred_mh = experiment_results
                all_results = mimo_mh_results + mimo_results
        
        elif user_choice == 4:  # SISO + MIMO + MIMO-MH-WLS
            print(f"\n{'='*60}")
            print("EXECUTION: SISO + MIMO + MIMO-MH-WLS")
            
            # SISO
            print(f"\n{'='*60}")
            print("PREDICTION WITH SISO")
            siso_results = run_siso_experiments(input_matrix, num_rows, num_outputs)
            
            # MIMO
            print(f"\n{'='*60}")
            print("PREDICTION WITH MIMO")
            mimo_results, Y_test_mimo, Y_pred_mimo = run_mimo_single_horizon_experiments(input_matrix, num_rows, num_outputs)
            
            # MIMO-MH avec WLS uniquement
            if CONFIG['reconciliation']:
                print(f"\n{'='*60}")
                print("PREDICTION WITH MIMO-MH-WLS")
                experiment_results = run_mimo_multihorizon_experiment(input_matrix, num_rows, num_outputs)
                mimo_mh_results, mimo_mh_wls_results, Y_test_mh, Y_pred_mh, Y_pred_reconciled = experiment_results
                all_results = siso_results + mimo_results + mimo_mh_wls_results
            else:
                print("WARNING: WLS reconciliation is not enabled in configuration.")
                all_results = siso_results + mimo_results
        
        elif user_choice == 5:  # MIMO-MH-WLS only and persistence
            if CONFIG['reconciliation']:
                print(f"\n{'='*60}")
                print("EXÉCUTION: MIMO-MH-WLS ONLY")
                
                experiment_results = run_mimo_multihorizon_experiment(input_matrix, num_rows, num_outputs)
                mimo_mh_results, mimo_mh_wls_results, Y_test_mh, Y_pred_mh, Y_pred_reconciled = experiment_results
                # Persistence
                pers_results = run_persistence_models(input_matrix, num_rows, num_outputs)  
                all_results = mimo_mh_wls_results + pers_results
            else:
                print("ERROR: WLS reconciliation is not enabled in configuration.")
                print("Cannot run MIMO-MH-WLS.")
                return
        
        elif user_choice == 6:  # MIMO-MH only and persistence
            print(f"\n{'='*60}")
            print("EXECUTION: MIMO-MH ONLY")
            experiment_results = run_mimo_multihorizon_experiment(input_matrix, num_rows, num_outputs)
            # Persistence
            pers_results = run_persistence_models(input_matrix, num_rows, num_outputs)  
            if CONFIG['reconciliation']:
                mimo_mh_results, mimo_mh_wls_results, Y_test_mh, Y_pred_mh, Y_pred_reconciled = experiment_results
                all_results = mimo_mh_results + pers_results
            else:
                mimo_mh_results, Y_test_mh, Y_pred_mh = experiment_results
                all_results = mimo_mh_results + pers_results
        
        elif user_choice == 7:  # MIMO only and persistence
            print(f"\n{'='*60}")
            print("EXECUTION: MIMO ONLY")
            # Persistence
            pers_results = run_persistence_models(input_matrix, num_rows, num_outputs)  
            
            mimo_results, Y_test_mimo, Y_pred_mimo = run_mimo_single_horizon_experiments(input_matrix, num_rows, num_outputs)
            all_results = mimo_results + pers_results
        
        elif user_choice == 8:  # SISO only and persistence
            print(f"\n{'='*60}")
            print("EXECUTION: SISO ONLY")
            
            siso_results = run_siso_experiments(input_matrix, num_rows, num_outputs)
            pers_results = run_persistence_models(input_matrix, num_rows, num_outputs)
            all_results = siso_results + pers_results
        
        # Create results DataFrame
        if all_results:
            df_results = pd.DataFrame(
                all_results, 
                columns=['Model', 'Horizon', 'Variable', 'nRMSE', 'nMAE', 'nMBE', 'R2']
            )
            
            # Generate plots depending on executed models
            models_executed = df_results['Model'].unique()
            
            print(f"\nExecuted models: {', '.join(models_executed)}")
            
            # General comparison plot
            plot_model_comparison(df_results)
            
            # Graphiques spécifiques par modèle
            for model in models_executed:
                if model in ['MIMO-MH-WLS', 'MIMO-MH', 'MIMO', 'SISO']:
                    plot_metrics_by_horizon(df_results, model)
                    plot_heatmaps(df_results[df_results['Model'] == model])
            
            # Graphique de comparaison MIMO si les données sont disponibles
            '''
            if Y_test_mh is not None and Y_pred_mh is not None:
                plot_combined_mimo_comparison_advanced(mimo_mh_results, Y_test_mh, Y_pred_mh, num_outputs, selected_horizons=[1, 6, 12, 18, 24])
            '''
            # Display detailed results
            print("\n=== DETAILED TABLE ===")
            print(df_results.to_string(index=False))
        else:
            print("No results to display.")
        
        # Temps d'exécution
        execution_time = time.time() - start_time
        print(f"\nTotal execution time: {execution_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":

    results = main()
