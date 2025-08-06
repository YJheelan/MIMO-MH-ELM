# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 16:24:14 2025

@author: yjheelan
"""

"""
Visualization Module for Model Results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import CONFIG

def compare_models_performance(df_results):
    """Compares performance between MIMO-MH and MIMO-MH-WLS models"""
    print("\n=== PERFORMANCE COMPARISON ===")
    # Moyennes globales par modèle
    comparison = df_results.groupby('Model')[['nRMSE', 'nMAE', 'R2']].mean()
    print("Average performance:")
    print(comparison)
    
    # Amélioration apportée par WLS
    if 'MIMO-MH-WLS' in df_results['Model'].values:
        mimo_mh = df_results[df_results['Model'] == 'MIMO-MH'][['nRMSE', 'nMAE', 'R2']].mean()
        mimo_mh_wls = df_results[df_results['Model'] == 'MIMO-MH-WLS'][['nRMSE', 'nMAE', 'R2']].mean()
        
        improvement = {
            'nRMSE': ((mimo_mh['nRMSE'] - mimo_mh_wls['nRMSE']) / mimo_mh['nRMSE']) * 100,
            'nMAE': ((mimo_mh['nMAE'] - mimo_mh_wls['nMAE']) / mimo_mh['nMAE']) * 100,
            'R2': ((mimo_mh_wls['R2'] - mimo_mh['R2']) / mimo_mh['R2']) * 100
        }
        
        print(f"\nWLS Reconciliation Gains:")
        print(f"- nRMSE reduction: {improvement['nRMSE']:.2f}%")
        print(f"- nMAE reduction: {improvement['nMAE']:.2f}%")
        print(f"- R² improvement: {improvement['R2']:.2f}%")


def plot_model_comparison(df_results):
    """Comparison plot of models by variable for different metrics"""
    output_names = CONFIG['output_names']
    metrics = {
        'nRMSE': {'title': 'nRMSE', 'ylabel': 'Normalized nRMSE', 'suptitle': 'nRMSE comparison across models'},
        'R2': {'title': 'R²', 'ylabel': 'R² (coefficient of determination)', 'suptitle': 'R² comparison across models'},
        'nMAE': {'title': 'nMAE', 'ylabel': 'Normalized nMAE', 'suptitle': 'nMAE comparison across models'},
        'nMBE': {'title': 'nMBE', 'ylabel': 'Normalized nMBE', 'suptitle': 'nMBE comparison across models'}
    }
    
    def create_subplot_for_metric(metric_name, metric_config):
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 12), sharex=True)
        axes = axes.flatten()
        
        # Define offset for MIMO model
        mimo_offset = 0 #offset 
        
        for i, var in enumerate(output_names):
            ax = axes[i]
            for model in df_results['Model'].unique():
                subset = df_results[(df_results['Variable'] == var) & (df_results['Model'] == model)]
                if not subset.empty:
                    # Apply offset only for MIMO model
                    y_values = subset[metric_name]
                    if model == 'MIMO':
                        y_values = y_values + mimo_offset
                        # Add the offset information to the legend
                        label = f"{model} (offset +{mimo_offset})"
                    else:
                        label = model
                    
                    ax.plot(subset['Horizon'], y_values, marker='o', label=label, linewidth=2)
            
            ax.set_title(f"{metric_config['title']} for {var}")
            ax.set_xlabel("Horizon (hours)")
            ax.set_ylabel(metric_config['ylabel'])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(metric_config['suptitle'], fontsize=16, y=1.02)
        plt.show()
        plt.close()
    
    for metric_name, metric_config in metrics.items():
        print(f"Generating plot for {metric_name}...")
        create_subplot_for_metric(metric_name, metric_config)

'''
def plot_model_comparison_mean(df_results):
    """Graphique de comparaison des modèles"""
    metrics = ['nRMSE', 'R2', 'nMAE', 'nMBE']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for model in df_results['Model'].unique():
            model_data = df_results[df_results['Model'] == model]
            avg_by_horizon = model_data.groupby('Horizon')[metric].mean()
            
            ax.plot(avg_by_horizon.index, avg_by_horizon.values, 
                   marker='o', label=model, linewidth=2, markersize=6)
        
        ax.set_title(f'Comparaison {metric} par horizon')
        ax.set_xlabel('Horizon (heures)')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Moyenne des variables énergétique', fontsize=16, y=1.02)
    plt.show()
'''

def plot_metrics_by_horizon(df_results, model_name):
    """Plot of metrics by horizon for a given model"""
    output_names = CONFIG['output_names']
    subset = df_results[df_results['Model'] == model_name]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    for var in output_names:
        var_data = subset[subset['Variable'] == var]
        if not var_data.empty:
            ax1.plot(var_data['Horizon'], var_data['nRMSE'], marker='o', linewidth=2, markersize=4, label=var)
            ax2.plot(var_data['Horizon'], var_data['R2'], marker='o', linewidth=2, markersize=4, label=var)
            ax3.plot(var_data['Horizon'], var_data['nMAE'], marker='o', linewidth=2, markersize=4, label=var)
            ax4.plot(var_data['Horizon'], var_data['nMBE'], marker='o', linewidth=2, markersize=4, label=var)
    
    for ax, title, ylabel in zip([ax1, ax2, ax3, ax4], 
                                 ['nRMSE evolution by horizon', 'R² evolution by horizon', 
                                  'nMAE evolution by horizon', 'nMBE evolution by horizon'],
                                 ['nRMSE', 'R²', 'nMAE', 'nMBE']):
        ax.set_xlabel('Forecast horizon (h)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'Performance metrics - {model_name}', fontsize=16, y=1.02)
    plt.show()

def plot_heatmaps(df_results):
    """
    Displays heatmaps for nRMSE, nMAE, and R² metrics for a given model.

    Args:
        df_results (pd.DataFrame): Model results including columns:
            ['Model', 'Horizon', 'Variable', 'nRMSE', 'nMAE', 'nMBE', 'R2']
    """
    required_columns = {'Model', 'Horizon', 'Variable', 'nRMSE', 'nMAE', 'R2'}
    missing = required_columns - set(df_results.columns)
    if missing:
        raise ValueError(f"Missing columns in df_results: {missing}")
    
    model_name = df_results['Model'].unique()
    if len(model_name) == 1:
        model_name = model_name[0]
    else:
        model_name = ', '.join(model_name)

    print(f"Creating detailed heatmaps for model: {model_name}")
    print("=" * 60)

    # Création des tables pivot pour chaque métrique
    def create_pivot(metric):
        return df_results.pivot_table(index='Variable', columns='Horizon', values=metric, aggfunc='mean')

    pivot_nrmse = create_pivot('nRMSE')
    pivot_r2 = create_pivot('R2')
    pivot_nmae = create_pivot('nMAE')

    # Création des heatmaps
    fig, axes = plt.subplots(3, 1, figsize=(20, 16))
    fig.suptitle(f"Results for model: {model_name}", fontsize=18, fontweight='bold')

    heatmap_cfgs = [
        (pivot_nrmse, 'nRMSE', 'YlOrRd', axes[0]),
        (pivot_r2, 'R²', 'Blues', axes[1]),
        (pivot_nmae, 'nMAE', 'Greens', axes[2]),
    ]

    for pivot, label, cmap, ax in heatmap_cfgs:
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap, ax=ax, cbar_kws={'label': label})
        ax.set_title(f'{label} by Variable and Forecast Horizon')
        ax.set_xlabel('Horizon (h)')
        ax.set_ylabel('Variable')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    

def print_best_results(df_results):
    """
    Displays the best results by variable based on R².

    Args:
        df_results (pd.DataFrame): Model results
    """
    print(f"\n{'-'*60}")
    print("BEST RESULTS BY VARIABLE")
    print(f"{'-'*60}")
    
    best_horizons = df_results.loc[df_results.groupby("Variable")["R2"].idxmax()]
    print(best_horizons[["Variable", "Model", "Horizon", "R2", "nRMSE"]])
    print("\n")
    
    
def plot_combined_mimo_comparison_advanced(results_mh,
                                          Y_test_mh, Y_pred_mh, num_outputs, selected_horizons=None):

#def plot_combined_mimo_comparison_advanced(results_mimo, results_mh, Y_test_mimo, Y_pred_mimo, 
#                                          Y_test_mh, Y_pred_mh, num_outputs, selected_horizons=None):
    """
    Advanced version with displayed metrics for MIMO-MH forecasts.

    Args:
        results_mh (list): Results from MIMO-MH experiments
        Y_test_mh (np.ndarray): Ground truth values for MIMO-MH
        Y_pred_mh (np.ndarray): Predicted values for MIMO-MH
        num_outputs (int): Number of output variables
        selected_horizons (list, optional): Specific horizons to plot (default: [1, 6, 12, 18, 24])
    """
    output_names = CONFIG['output_names']
    max_horizon = CONFIG['max_horizon']
    
    #Define the horizons to be traced
    if selected_horizons is None:
        selected_horizons = [1, 6, 12, 18, 24]
    
    # Filter valid horizons
    valid_horizons = [h for h in selected_horizons if h <= max_horizon]
    
    for horizon in valid_horizons:
        h = horizon - 1  # Index (0-based)
        fig, axes = plt.subplots(num_outputs, 1, figsize=(16, 4 * num_outputs), sharex=True)
        if num_outputs == 1:
            axes = [axes]
        
        for i in range(num_outputs):
            ax = axes[i]
            
            # Limit display
            max_points_mh = min(100, len(Y_test_mh))
            '''
            max_points_mimo = min(100, len(Y_test_mimo))
            '''
            # Plotting data
            ax.plot(Y_test_mh[:max_points_mh, h, i], 'g-', linewidth=2.5, 
                   label='Valeurs réelles MIMO-MH', alpha=0.9)
            '''
            # MIMO
            if Y_pred_mimo.ndim == 3:
                ax.plot(Y_pred_mimo[:max_points_mimo, h, i], 'b--', linewidth=2, 
                       label='MIMO-ELM', alpha=0.8)
                ax.plot(Y_test_mimo[:max_points_mimo, h, i], 'c-', linewidth=2, 
                       label='Valeurs réelles MIMO', alpha=0.8)
            else:
                ax.plot(Y_pred_mimo[:max_points_mimo, i], 'b--', linewidth=2, 
                       label='MIMO-ELM', alpha=0.8)
                ax.plot(Y_test_mimo[:max_points_mimo, i], 'c-', linewidth=2, 
                       label='Valeurs réelles MIMO', alpha=0.8)
            '''
            # MIMO-MH
            ax.plot(Y_pred_mh[:max_points_mh, h, i], 'y:', linewidth=2.5, 
                   label='MIMO-MH', alpha=0.8)
            
            # Récupérer les métriques depuis les résultats
            mimo_metrics = None
            mh_metrics = None
            
            
            # Chercher les métriques pour cet horizon et cette variable
            """
            for result in results_mimo:
                if result[1] == horizon and result[2] == output_names[i]:
                    mimo_metrics = result[3:7]  # [nrmse, nmae, nmbe, r2]
                    break
            """
            for result in results_mh:
                if result[1] == horizon and result[2] == output_names[i]:
                    mh_metrics = result[3:7]  # [nrmse, nmae, nmbe, r2]
                    break
            
            # Créer le texte des métriques
            metrics_text = ""
            '''
            if mimo_metrics:
                metrics_text += f"MIMO - NRMSE: {mimo_metrics[0]:.3f}, nMAE: {mimo_metrics[1]:.3f}, nMBE: {mimo_metrics[2]:.3f}, R²: {mimo_metrics[3]:.3f}\n"
            '''
            if mh_metrics:
                metrics_text += f"MIMO - NRMSE: {mh_metrics[0]:.3f}, nMAE: {mh_metrics[1]:.3f}, nMBE: {mh_metrics[2]:.3f}, R²: {mh_metrics[3]:.3f}\n"
            
            # Afficher les métriques
            if metrics_text:
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                       fontsize=9, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Configuration des axes
            ax.set_xlim([0, max_points_mh])
            ax.set_title(f"Horizon {horizon}h - Variable : {output_names[i]}", 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (h)', fontsize=10)
            ax.set_ylabel(output_names[i], fontsize=10)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Améliorer l'apparence
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        fig.suptitle(f"Forecast - MIMO-MH - Horizon {horizon}h", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()