# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 16:32:51 2025

@author: yjheelan
"""

import numpy as np
import pandas as pd
from scipy.linalg import pinv
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import time

# Configuration globale
CONFIG = {
    'data_file': "Data.csv",
    'window_size': 48,
    'num_hidden': 1000,
    'num_initializations': 1,
    'max_horizon': 24,
    'train_ratio': 0.8,
    'lambda_reg': 1e-6,  # régularisation Ridge
    'output_names': ['Total_MW', 'Thermal_MW', 'Hydro_MW', 'Micro_Hydro_MW', 
                     'Solar_MW', 'Wind_MW', 'BioEner_MW', 'Import_MW'],
    'energy_cols': [2, 3, 5, 6, 7, 8, 9, 11],
    'reconciliation': True,  # Activation de la réconciliation WLS
}

def load_and_preprocess_data(file_path):
    """
    Purpose:
    - Loads and preprocesses the dataset for forecasting.
    Arguments:
    - data_file (str): Path to the CSV file
    What it does:
    - Reads the CSV file
    - Converts the time column into datetime format
    - Creates a feature matrix including:
        - Hour of the day (encoded as sine/cosine)
        - Day of the week
        - Rolling averages (optional)
        - Filters out rows with negative target values
    Returns:
    - X (np.ndarray): Feature matrix
    - y (np.ndarray): Target matrix (for each energy variable)
    - columns (list): Names of features for reference
    Usage:
    - Used as the initial data input to train and test the forecasting models.
    """
    print("Chargement et prétraitement des données...")
    
    # Charger les données
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    # Traitement des dates
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df.index = df["Date"]
    df.drop(['Date'], axis=1, inplace=True)
    df = df.fillna(0)
    # Extraction des heures et transformation cyclique
    Hours = df.index.hour + 1  # de 1 à 24
    
    Hours_sin = np.sin(2 * np.pi * Hours / 24)
    Hours_cos = np.cos(2 * np.pi * Hours / 24)
    
    # Extraction des données d'énergie
    energy_data = np.column_stack([
        np.maximum(df.iloc[:, i].values, 0) for i in CONFIG['energy_cols']
    ])
    
    # Création de la matrice d'entrée
    input_matrix = np.column_stack([energy_data, Hours_sin, Hours_cos])
    
    num_rows = input_matrix.shape[0]
    num_outputs = len(CONFIG['output_names'])
    print(f"Données chargées: {num_rows} lignes, {input_matrix.shape[1]} colonnes")
    
    return input_matrix, num_rows, num_outputs

def sliding_window_mimo(input_matrix, pred_horizon, window_size, num_rows ):
    '''    
    """
    Crée des fenêtres glissantes pour prédiction à horizon unique
    
    Args:
        X (np.array): Matrice d'entrée
        pred_horizon (int): Horizon de prédiction
        window_size (int): Taille de la fenêtre
        num_rows (int): Nombre de lignes dans les données
        
    Returns:
        tuple: (X_windowed, Y_target)
    """
    obs = num_rows - window_size - pred_horizon
    print("sliding_window_mimo :")
    print("fenêtre d'observation",obs)
    X_windowed = np.lib.stride_tricks.sliding_window_view(X, (window_size, X.shape[1]))[:obs]
    X_windowed = X_windowed.reshape(obs, -1)
    Y_target = X[window_size + pred_horizon - 1 : window_size + pred_horizon - 1 + obs, :-2]


    print("len(X_windowed)",len(X_windowed))
    print("len(Y_target)",len(Y_target))
    return np.nan_to_num(X_windowed), np.nan_to_num(Y_target)
    '''
    
    obs = num_rows - window_size - pred_horizon
    X = np.zeros((obs, window_size * input_matrix.shape[1])) #(Entrée avec fenêtres de window_size points)
    Y = np.zeros((obs, input_matrix.shape[1] - 2))  # Sorties sans sinus/cosinus
    for i in range(obs):
        window_data = input_matrix[i:i+window_size, :].T
        X[i, :] = window_data.flatten()  # pour aplatir la fenêtre en une seule ligne
        # ce qu'on veut est la production d'énergie prediction_horizon dans le futur
        Y[i, :] = input_matrix[i+window_size+pred_horizon-1, :-2]  # Toutes les colonnes sauf sin/cos
    return np.nan_to_num(X), np.nan_to_num(Y)

def split_train_test(X, Y, train_ratio=0.8):
    """
    Divise les données en ensembles d'entraînement et de test
    
    Args:
        X (np.array): Données d'entrée
        Y (np.array): Données de sortie
        train_ratio (float): Proportion pour l'entraînement
        
    Returns:
        tuple: (X_train, Y_train, X_test, Y_test)
    """
    trainSize = round(0.8 * X.shape[0])
    X_train = X[:trainSize, :]
    Y_train = Y[:trainSize, :]
    X_test = X[trainSize:, :]
    Y_test = Y[trainSize:, :]
    '''
    print("split_train_test :")
    print("train size", train_size)
    print("len(X_train)", len(X_train))
    print("len(Y_train)", len(Y_train))
    print("len(X_test)",len(X_test))
    print(" len(Y_test)", len(Y_test))
    '''
    return X_train, Y_train, X_test, Y_test

def train_elm(X_train, Y_train, X_test, Y_test, num_hidden=None, num_initializations=None):
    """
    Entraîne un modèle ELM avec multiple initialisations
    
    Args:
        X_train (np.array): Données d'entraînement (entrées)
        Y_train (np.array): Données d'entraînement (sorties)
        X_test (np.array): Données de test (entrées)
        Y_test (np.array): Données de test (sorties)
        num_hidden (int): Nombre de neurones cachés
        num_initializations (int): Nombre d'initialisations
        
    Returns:
        tuple: (W, b, beta) - Paramètres du meilleur modèle
    """
    if num_hidden is None:
        num_hidden = CONFIG['num_hidden']
    if num_initializations is None:
        num_initializations = CONFIG['num_initializations']
    
    best_rmse = float('inf')
    best_model = None
    best_inputWeights = None  # Initialiser les meilleurs poids d'entrée
    best_bias = None  # Initialiser le meilleur biais
    best_outputWeights = None  # Initialiser les meilleurs poids de sortie
    
    for init in range(num_initializations):
        # Initialisation aléatoire des poids
        W = np.random.rand(num_hidden, X_train.shape[1]) #inputWeights
        b = np.random.rand(num_hidden, 1) #bias
        
        # Calcul de la couche cachée (activation ReLU)
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
        # Prédiction sur l'ensemble de test
        H_test = np.maximum(0, X_test @ W.T + b.T)
        Y_pred = H_test @ beta # Prédictions non normalisées
        Y_pred = np.maximum(Y_pred, 0)
        
        # Calcul du RMSE
        rmse = np.mean(np.sqrt(np.mean(( Y_test - Y_pred)**2, axis=0)))
        
        # Sauvegarde du meilleur modèle
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
    Effectue des prédictions avec un modèle ELM entraîné
    
    Args:
        X (np.array): Données d'entrée
        model (tuple): (W, b, beta) - Paramètres du modèle
        
    Returns:
        np.array: Prédictions
    """
    W, b, beta = model
    H = np.maximum(0, X @ W.T + b.T)
    Y_pred = np.maximum(H @ beta, 0)
    return Y_pred

def get_metrics(y_true, y_pred):
    """
    Calcule les métriques d'évaluation
    
    Args:
        y_true (np.array): Valeurs vraies
        y_pred (np.array): Prédictions
        
    Returns:
        tuple: (nrmse, nmae, nmbe, r2)
    """
    # Gestion des dimensions
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Calcul des métriques de base
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    mbe = np.mean(y_true - y_pred, axis=0)
    mean_y = np.mean(y_true, axis=0)
    
    # Normalisation et autres métriques
    with np.errstate(divide='ignore', invalid='ignore'):
        nrmse = np.nan_to_num(rmse / mean_y)
        nmae = np.nan_to_num(mae / mean_y)
        nmbe = np.nan_to_num(mbe / mean_y)
        
        # Coefficient de détermination R²
        ss_res = np.sum((y_true - y_pred)**2, axis=0)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
        r2 = 1 - (ss_res / ss_tot)
        r2 = np.nan_to_num(r2)
    
    return nrmse, nmae, nmbe, r2

def run_mimo_single_horizon_experiments(input_matrix, num_rows, num_outputs):
    """
    Exécute les expériences pour chaque horizon individuellement
    
    Args:
        input_matrix (np.array): Matrice d'entrée
        num_rows (int): Nombre de lignes
        num_outputs (int): Nombre de sorties
        
    Returns:
        list: Résultats des expériences
    """
    results = []
    output_names = CONFIG['output_names']
    max_horizon = CONFIG['max_horizon']
    
    print(f"\n{'='*60}")
    print("EXPÉRIENCES MIMO : HORIZON SIMPLE")
    
    for horizon in range(1, max_horizon + 1):
        print(f"Traitement horizon {horizon}h...")
        
        # Préparation des données
        X_single, Y_single = sliding_window_mimo(
            input_matrix, horizon, CONFIG['window_size'], num_rows
        )
        X_train, Y_train, X_test, Y_test = split_train_test(
            X_single, Y_single, CONFIG['train_ratio']
        )
        
        # Entraînement
        model = train_elm(X_train, Y_train, X_test, Y_test)
        
        # Prédiction
        Y_pred = predict_elm(X_test, model)
        
        # Évaluation par variable
        for i in range(num_outputs):
            nrmse, nmae, nmbe, r2 = get_metrics(Y_test[:, i], Y_pred[:, i])
            results.append([
                'MIMO', horizon, output_names[i], 
                float(nrmse), float(nmae), float(nmbe), float(r2)
            ])
    
    return results, Y_test, Y_pred


# PERS
def run_persistence_models(input_matrix, num_rows, num_outputs):
    """
    Calcule les métriques pour deux modèles de persistance :
      - 'Persistance' : reprend à l’horizon h
      - 'Persistance-24h' : reprend à 24 h en arrière, quel que soit h
    Retourne une liste structurée comme run_mimo_multihorizon_experiment
    """
    results = []
    output_names = CONFIG['output_names']
    window_size = CONFIG['window_size']
    max_horizon = CONFIG['max_horizon']
    train_ratio = CONFIG['train_ratio']
    print(f"\n{'='*60}")
    print("Calcul des performances des persistance (horizon et 24h)…")

    for horizon in range(1, max_horizon + 1):
        print(f" → Persistance horizon {horizon}h")
        # Extraire uniquement Y (pas besoin de X pour la persistance)
        num_obs = num_rows - window_size - horizon
        Y = np.zeros((num_obs, input_matrix.shape[1] - 2))
        for i in range(num_obs):
            Y[i, :] = input_matrix[i + window_size + horizon - 1, :-2]
        Y = np.nan_to_num(Y)

        # Séparation test
        train_size = int(train_ratio * Y.shape[0])
        Y_test = Y[train_size:, :]

        # Création des prédictions persistance
        Y_pers = np.zeros_like(Y_test)
        Y_pers_24h = np.zeros_like(Y_test)
        for i in range(Y_test.shape[0]):
            if i - horizon >= 0:
                Y_pers[i] = Y_test[i - horizon]
            # sinon reste à 0
            if i - 24 >= 0:
                Y_pers_24h[i] = Y_test[i - 24]

        # Calcul métriques par variable
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

# Graphe
def plot_model_comparison(df_results):
    """Graphique de comparaison des modèles par variable pour différentes métriques"""
    output_names = CONFIG['output_names']
    metrics = {
        'nRMSE': {'title': 'nRMSE', 'ylabel': 'nRMSE normalisé', 'suptitle': 'Comparaison du nRMSE de chaque modèle'},
        'R2': {'title': 'R²', 'ylabel': 'R² (coefficient de détermination)', 'suptitle': 'Comparaison du R² de chaque modèle'},
        'nMAE': {'title': 'nMAE', 'ylabel': 'nMAE normalisé', 'suptitle': 'Comparaison du nMAE de chaque modèle'},
        'nMBE': {'title': 'nMBE', 'ylabel': 'nMBE normalisé', 'suptitle': 'Comparaison du nMBE de chaque modèle'}
    }
    
    def create_subplot_for_metric(metric_name, metric_config):
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 12), sharex=True)
        axes = axes.flatten()
        
        # Définir l'offset pour le modèle MIMO
        mimo_offset = 0 #offset 
        
        for i, var in enumerate(output_names):
            ax = axes[i]
            for model in df_results['Model'].unique():
                subset = df_results[(df_results['Variable'] == var) & (df_results['Model'] == model)]
                if not subset.empty:
                    # Appliquer l'offset uniquement pour le modèle MIMO
                    y_values = subset[metric_name]
                    if model == 'MIMO':
                        y_values = y_values + mimo_offset
                        # Ajouter l'information de l'offset dans la légende
                        label = f"{model} (offset +{mimo_offset})"
                    else:
                        label = model
                    
                    ax.plot(subset['Horizon'], y_values, marker='o', label=label, linewidth=2)
            
            ax.set_title(f"{metric_config['title']} pour {var}")
            ax.set_xlabel("Horizon (heures)")
            ax.set_ylabel(metric_config['ylabel'])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(metric_config['suptitle'], fontsize=16, y=1.02)
        plt.show()
        plt.close()
    
    for metric_name, metric_config in metrics.items():
        print(f"Génération du graphique pour {metric_name}...")
        create_subplot_for_metric(metric_name, metric_config)

def plot_metrics_by_horizon(df_results, model_name):
    """Graphique des métriques par horizon pour un modèle donné"""
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
                                ['Évolution du nRMSE par horizon', 'Évolution du R² par horizon', 
                                 'Évolution du nMAE par horizon', 'Évolution du nMBE par horizon'],
                                ['nRMSE', 'R²', 'nMAE', 'nMBE']):
        ax.set_xlabel('Horizon de prédiction (h)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'Métriques de performance - {model_name}', fontsize=16, y=1.02)
    plt.show()

def plot_heatmaps(df_results):
    """
    Affiche des heatmaps des métriques nRMSE, nMAE et R² pour un modèle donné.
    
    Args:
        df_results (pd.DataFrame): Résultats contenant les colonnes :
            ['Model', 'Horizon', 'Variable', 'nRMSE', 'nMAE', 'nMBE', 'R2']
    """
    required_columns = {'Model', 'Horizon', 'Variable', 'nRMSE', 'nMAE', 'R2'}
    missing = required_columns - set(df_results.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans df_results : {missing}")
    
    model_name = df_results['Model'].unique()
    if len(model_name) == 1:
        model_name = model_name[0]
    else:
        model_name = ', '.join(model_name)

    print(f"Création des heatmaps des résultats détaillés pour le modèle : {model_name}")
    print("=" * 60)

    # Création des tables pivot pour chaque métrique
    def create_pivot(metric):
        return df_results.pivot_table(index='Variable', columns='Horizon', values=metric, aggfunc='mean')

    pivot_nrmse = create_pivot('nRMSE')
    pivot_r2 = create_pivot('R2')
    pivot_nmae = create_pivot('nMAE')

    # Création des heatmaps
    fig, axes = plt.subplots(3, 1, figsize=(20, 16))
    fig.suptitle(f"Résultats pour le modèle : {model_name}", fontsize=18, fontweight='bold')

    heatmap_cfgs = [
        (pivot_nrmse, 'nRMSE', 'YlOrRd', axes[0]),
        (pivot_r2, 'R²', 'Blues', axes[1]),
        (pivot_nmae, 'nMAE', 'Greens', axes[2]),
    ]

    for pivot, label, cmap, ax in heatmap_cfgs:
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap, ax=ax, cbar_kws={'label': label})
        ax.set_title(f'{label} par Variable et Horizon de Prédiction')
        ax.set_xlabel('Horizon (h)')
        ax.set_ylabel('Variable')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
def main():
    print("PRÉDICTION AVEC MIMO-MH et RÉCONCILIATION WLS")
    start_time = time.time()
    try:
        input_matrix, num_rows, num_outputs = load_and_preprocess_data(CONFIG['data_file'])
        
        pers_results = run_persistence_models(input_matrix, num_rows, num_outputs)  
        
        mimo_results, Y_test_mimo, Y_pred_mimo = run_mimo_single_horizon_experiments(input_matrix, num_rows, num_outputs)
        all_results = mimo_results + pers_results
        
        df_results = pd.DataFrame(
            all_results, 
            columns=['Model', 'Horizon', 'Variable', 'nRMSE', 'nMAE', 'nMBE', 'R2']
        )
        
        print("\n=== TABLEAU DÉTAILLÉ ===")
        print(df_results.to_string(index=False))

        # Graphiques
        plot_model_comparison(df_results)
        plot_metrics_by_horizon(df_results, 'MIMO')
        
        # Temps d'exécution
        execution_time = time.time() - start_time
        print(f"Temps d'exécution total: {execution_time:.2f} secondes")
        
    except Exception as e:
        print(f"Erreur lors de l'exécution: {str(e)}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    results = main()