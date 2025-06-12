# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:31:04 2025

@author: yjheelan
"""

import pandas as pd
import numpy as np
#import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import time # pour la mesure du temps d'exécution
from scipy.linalg import pinv #  Pseudo-inverse de Moore-Penrose pour la régression
import scipy as sp
mpl.rcParams['figure.dpi'] = 300
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
#---------------------------------------------------------------------------
#Début
# Mesurer le temps d'exécution
start_time = time.time()
# import du fichier data
dfData = pd.read_csv("Data.csv")

#Supprime les espaces au début et à la fin de tous les noms de colonnes du DataFrame.
dfData.columns = dfData.columns.str.strip() 
# Définit la colonne "Date" comme index du DataFrame dfData, en la convertissant en format datetime.
dfData["Date"] = pd.to_datetime(dfData["Date"], utc=True) 
DateTime = pd.to_datetime(dfData.iloc[:, 2], errors='coerce')
Hours = DateTime.dt.hour
dfData.index = dfData["Date"] #Date en index
# Une fois que "Date" a été copiée dans l'index, on supprime la colonne "Date" du DataFrame (car en trop).
dfData.drop(['Date'], axis=1, inplace=True) 
# Optimization Parameters --------------------------------------------------
window_size = 48  # Number of rows for an observation
numHiddenUnits = 1000 # Number of hidden neurons (4096)
numInitializations = 1  # Number of initializations to try

#---------------------------------------------------------------------------
#Extract necessary columns
## On utilise np.maximum(valeur, 0) pour nous assurer que toutes les valeurs sont positives
TotalProduction_MW = np.maximum(dfData.iloc[:, 2].values, 0)
Thermal_MW = np.maximum(dfData.iloc[:, 3].values, 0)
Hydro_MW = np.maximum(dfData.iloc[:, 5].values, 0)
Solar_MW = np.maximum(dfData.iloc[:, 7].values, 0)
Wind_MW = np.maximum(dfData.iloc[:, 8].values, 0)
BioEnergy_MW = np.maximum(dfData.iloc[:, 9].values, 0)
Import_MW = np.maximum(dfData.iloc[:, 11].values, 0)
#---------------------------------------------------------------------------
# Transform the hour (0 to 23) with an offset
Hours_offset = Hours + 1  # Add offset to ensure positive values
## Encoder l'heure de la journée avec des fonctions sinus et cosinus (encodage cyclique)
Hours_sin = np.sin(2 * np.pi * Hours_offset / 24)
Hours_cos = np.cos(2 * np.pi * Hours_offset / 24)
## matrice contenant toutes les variables (Combine toutes les variables (7 énergétiques + 2 temporelles) en une seule matrice)
input_matrix = np.column_stack([
    TotalProduction_MW, Thermal_MW, Hydro_MW, Solar_MW, 
    Wind_MW, BioEnergy_MW, Import_MW, Hours_sin, Hours_cos
])
# Les sorties-----------------------------------------------------------------
outputNames = ['Total_MW', 'Thermal_MW', 'Hydro_MW', 
               'Solar_MW', 'Wind_MW', 'BioEner_MW', 'Import_MW']
# Résumé Initialisation---------------------------------------------------------------------------
"""  
window_size = 48  # Number of rows for an observation
numHiddenUnits = 1000 # Number of hidden neurons (4096)
numInitializations = 1  # Number of initializations to try
Entrées : "Production totale (MW)", "Thermique (MW)", "Hydraulique (MW)", "Micro-hydraulique (MW)", "Solaire photovoltaïque (MW)", "Eolien (MW)", "Bioénergies (MW)", "Importations (MW), Hours_sin, Hours_cos"
Sorties : 'Total_MW', 'Thermal_MW', 'Hydro_MW', 'Solar_MW', 'Wind_MW', 'BioEner_MW', 'Import_MW'

Pour le modèle MIMO-ELM, la paramétrisation est la suivante :
    Entrées : 7 sources énergétiques observées sur 48 heures et 2 composantes temporelles, soit 338 entrées;
    Couche cachée : 1000 neurones;
    Sorties : 7 * 24 = 168;
    Horizon de prévision : 24 heure par pas horaire (par run).

"""
#---------------------------------------------------------------------------
# On va faire une boucle qui va calculer pour chaque horizon de 1 à 24h les métriques + prévisions
# Initialise liste vide
all_results = []
horizon_metrics = {}
# listes pour les métriques de persistance 24h
horizon_metrics_24h = {}
all_results_24h = []
#♫ Début
print("Calcul des prédictions pour tous les horizons de 1 à 24 heures...")
# Fait une boucle pour le calcul des horizons de predictions de 1 à 24
for prediction_horizon in range(1, 25): # Prediction horizon (Prédit 1 pas de temps dans le futur)
    print(f"\nTraitement de l'horizon {prediction_horizon}h...") # Boucle de 1 à 24 pour chaque horizon...
    
    
    #---------------------------------------------------------------------------
    # Input-Output with Preallocation and windowed data (données en fenêtres glissantes)
    numRows = input_matrix.shape[0]
    numObservations = numRows - window_size - prediction_horizon
    
    ## Preallocate matrices X and Y
    X = np.zeros((numObservations, window_size * input_matrix.shape[1])) #(Entrée avec fenêtres de window_size points)
    Y = np.zeros((numObservations, input_matrix.shape[1] - 2))  # Sorties sans sinus/cosinus
    
    
    #---------------------------------------------------------------------------
    # Fill matrices X and Y
    for i in range(numObservations):
        # Pour chaque observation, on prend une fenêtre de données
        window_data = input_matrix[i:i+window_size, :].T
        X[i, :] = window_data.flatten()  # pour aplatir la fenêtre en une seule ligne
        # ce qu'on veut est la production d'énergie prediction_horizon dans le futur
        Y[i, :] = input_matrix[i+window_size+prediction_horizon-1, :-2]  # Toutes les colonnes sauf sin/cos
        
        #Chaque ligne de X contient 48 × 9 = 432 valeurs (48 points temporels × 9 variables)
        #Chaque ligne de Y contient les 7 variables énergétiques à prédire au temps t+1
    
    ## Remplacer les NaN par 0
    X = np.nan_to_num(X)
    Y = np.nan_to_num(Y)
    
    ## Prepare training and testing data
    trainSize = round(0.8 * X.shape[0])
    XTrain = X[:trainSize, :]
    YTrain = Y[:trainSize, :]
    XTest = X[trainSize:, :]
    YTest = Y[trainSize:, :]
    #---------------------------------------------------------------------------
    # ELM model parameters
    numOutputs = YTrain.shape[1] #sorties
    best_rmse = float('inf') # Initialiser le meilleur RMSE
    ##Poids d'entrée aléatoires (les poids entre l'entrée et la couche cachée sont générés aléatoirement)
    best_inputWeights = None  # Initialiser les meilleurs poids d'entrée
    best_bias = None  # Initialiser le meilleur biais
    best_outputWeights = None  # Initialiser les meilleurs poids de sortie
    
    #--------------------------------------------------------------------------
    # Tester plusieurs initialisations et calculs des métriques
    for init in range(numInitializations):
        ## Initialiser les poids de la couche cachée et le biais
        inputWeights = np.random.rand(numHiddenUnits, XTrain.shape[1])
        bias = np.random.rand(numHiddenUnits, 1)
        
        ## Calculer la sortie de la couche cachée  
        ## Fonction d'activation,  introduit la non-linéarité
        H = np.maximum(0, XTrain @ inputWeights.T + bias.T)
        
        ## Calculer les poids de sortie en utilisant la régression linéaire avec pseudo-inverse (pinv)
        outputWeights = pinv(H) @ YTrain # Régression corrigée
        
        ## Faire des prédictions sur l'ensemble de test
        H_test = np.maximum(0, XTest @ inputWeights.T + bias.T)
        YPred = H_test @ outputWeights # Prédictions non normalisées
        YPred = np.maximum(YPred, 0)   # S'assurer que les valeurs prédites sont non négatives
        
        
        
        #---------------------------------------------------------------------
        # Calculate error metrics
        rmse_values = np.zeros(numOutputs)
        for j in range(numOutputs):
            y_true = YTest[:, j]
            y_pred = YPred[:, j]
            rmse = np.sqrt(np.mean((y_pred - y_true)**2))
            rmse_values[j] = rmse
            
        ## Calculer le RMSE moyen sur toutes les sorties
        mean_rmse = np.mean(rmse_values)
        
        ## Keep the best initialization
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_inputWeights = inputWeights
            best_bias = bias
            best_outputWeights = outputWeights
            
    print(f"Best RMSE: {best_rmse} pour l'horizon {prediction_horizon}h...")
    
    
    #---------------------------------------------------------------------------
    # Prédictions finales avec les meilleurs poids
    H_final = np.maximum(0, XTest @ best_inputWeights.T + best_bias.T)
    YPred_final = H_final @ best_outputWeights
    
    ## S'assurer que les valeurs prédites sont non négatives
    YPred_final = np.maximum(YPred_final, 0)
    
    #---------------------------------------------------------------------------
    # Persistence, modèle i-horizon de prédiction
    YPersistence = np.zeros_like(YTest)
    for i in range(YTest.shape[0]):
        if i - prediction_horizon >= 0:
            YPersistence[i, :] = YTest[i - prediction_horizon, :]
        else:
            YPersistence[i, :] = 0

    # Persistence modèle 24h
    YPersistence_24h = np.zeros_like(YTest)
    for i in range(YTest.shape[0]):
        if i - 24 >= 0:  # Utilise toujours 24h en arrière
            YPersistence_24h[i, :] = YTest[i - 24, :]
        else:
            YPersistence_24h[i, :] = 0
    
    #---------------------------------------------------------------------------
    # Calculer les métriques pour les prédictions et les deux persistances
    results_for_horizon = []
    results_for_horizon_24h = []
    
    for j in range(numOutputs):
        y_true = YTest[:, j]
        y_pred = YPred_final[:, j]
        y_persist = YPersistence[:, j]
        y_persist_24h = YPersistence_24h[:, j]
        
        # Error metrics pour ELM
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        mae = np.mean(np.abs(y_pred - y_true))
        mbe = np.mean(y_pred - y_true)
        mean_y = np.mean(y_true)
        
        # Avoid division by zero
        if mean_y != 0:
            nrmse = rmse / mean_y
            nmae = mae / mean_y
            nmbe = mbe / mean_y
        else:
            nrmse = rmse
            nmae = mae
            nmbe = mbe
        
        # Persistence metrics
        rmse_persist = np.sqrt(np.mean((y_persist - y_true) ** 2))
        if mean_y != 0:
            nrmse_persist = rmse_persist / mean_y
        else:
            nrmse_persist = rmse_persist
        
        # Persistence 24h metrics
        rmse_persist_24h = np.sqrt(np.mean((y_persist_24h - y_true) ** 2))
        mae_persist_24h = np.mean(np.abs(y_persist_24h - y_true))
        mbe_persist_24h = np.mean(y_persist_24h - y_true)
        
        if mean_y != 0:
            nrmse_persist_24h = rmse_persist_24h / mean_y
            nmae_persist_24h = mae_persist_24h / mean_y
            nmbe_persist_24h = mbe_persist_24h / mean_y
        else:
            nrmse_persist_24h = rmse_persist_24h
            nmae_persist_24h = mae_persist_24h
            nmbe_persist_24h = mbe_persist_24h
        
        # Gain calculation (vs horizon persistence)
        if nrmse_persist != 0:
            gain = (nrmse_persist - nrmse) / nrmse_persist
        else:
            gain = 0

        # Gain calculation (vs 24h persistence)
        if nrmse_persist_24h != 0:
            gain_24h = (nrmse_persist_24h - nrmse) / nrmse_persist_24h
        else:
            gain_24h = 0
        
        # R2 pour ELM
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot != 0:
            r2 = 1 - (ss_res / ss_tot)
        else:
            r2 = 0

        # R2 pour persistence 24h #
        ss_res_24h = np.sum((y_true - y_persist_24h) ** 2)
        if ss_tot != 0:
            r2_persist_24h = 1 - (ss_res_24h / ss_tot)
        else:
            r2_persist_24h = 0
        
        # Ajouter aux résultats ELM (comparé à persistence originale = persistance horizon)
        results_for_horizon.append([
            prediction_horizon, outputNames[j], nrmse, gain, nmae, nmbe, r2
        ])

        # Ajouter aux résultats de persistence 24h
        results_for_horizon_24h.append([
            prediction_horizon, outputNames[j], nrmse_persist_24h, gain_24h, 
            nmae_persist_24h, nmbe_persist_24h, r2_persist_24h
        ])
        
    #---------------------------------------------------------------------------
    # Store results for this horizon
    all_results.extend(results_for_horizon)
    all_results_24h.extend(results_for_horizon_24h)
    
    ## Store average metrics for this horizon (ELM)
    horizon_avg_metrics = np.mean([[row[2], row[3], row[4], row[5], row[6]] 
                                   for row in results_for_horizon], axis=0)
    horizon_metrics[prediction_horizon] = {
        'nRMSE': horizon_avg_metrics[0],
        'Gain': horizon_avg_metrics[1],
        'nMAE': horizon_avg_metrics[2],
        'nMBE': horizon_avg_metrics[3],
        'R2': horizon_avg_metrics[4]
    }

    ## Store average metrics for this horizon (Persistence 24h)
    horizon_avg_metrics_24h = np.mean([[row[2], row[3], row[4], row[5], row[6]] 
                                       for row in results_for_horizon_24h], axis=0)
    horizon_metrics_24h[prediction_horizon] = {
        'nRMSE': horizon_avg_metrics_24h[0],
        'Gain': horizon_avg_metrics_24h[1],
        'nMAE': horizon_avg_metrics_24h[2],
        'nMBE': horizon_avg_metrics_24h[3],
        'R2': horizon_avg_metrics_24h[4]
    }
        
    #GRAPHE 1---------------------------------------------------------------------------
    # Tracer les prédictions pour chaque variable de sortie
    # DataFrame avec les noms de colonnes appropriés
    df_results = pd.DataFrame(results_for_horizon, columns=["Horizon", "Variable", "nRMSE", "Gain", "nMAE", "nMBE", "R2"])
    df_results_24h = pd.DataFrame(results_for_horizon_24h, columns=["Horizon", "Variable", "nRMSE", "Gain", "nMAE", "nMBE", "R2"])
    
    fig, axes = plt.subplots(numOutputs, 1, figsize=(15, 20), sharex=True)  # Plus large pour 3 courbes
    # Si il n'y a qu'une seule sortie, convertir en liste pour cohérence
    
    start_pers24 = 24  # Décalage de 24h pour la persistance24h
    if numOutputs == 1:
        axes = [axes]
    for j in range(numOutputs):
        ax = axes[j]
        ax.plot(YTest[:, j], 'b-', linewidth=2, label='Valeurs réelles', alpha=0.8)
        ax.plot(YPred_final[:, j], 'r--', linewidth=2, label='ELM', alpha=0.8)
        ax.plot(YPersistence[:, j], 'g:', linewidth=2, label='Persistence (horizon)', alpha=0.7)
       
        x_vals = range(start_pers24, len(YPersistence_24h[:, j]))
        ax.plot(x_vals, YPersistence_24h[start_pers24:, j], 'm-.', linewidth=2, label='Persistence 24h', alpha=0.7)
        
        ax.set_xlim([1, 100])
        
        # Récupérer les métriques à partir des DataFrames
        row_elm = df_results.iloc[j]
        row_24h = df_results_24h.iloc[j]
        
        ax.set_title(
            f"ELM - nRMSE: {row_elm['nRMSE']:.4f}, Gain: {row_elm['Gain']:.4f}, R²: {row_elm['R2']:.4f}\n"
            f"Pers24h - nRMSE: {row_24h['nRMSE']:.4f}, R²: {row_24h['R2']:.4f}",
            fontsize=9
        )
        ax.set_xlabel('Temps (h)')
        ax.set_ylabel(outputNames[j])
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Comparaison des modèles pour l'horizon {prediction_horizon}h", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

#TABLEAUX--------------------------------------------------------------------
# Create comprehensive results DataFrames + print results
columns = ["Horizon", "Variable", "nRMSE", "Gain", "nMAE", "nMBE", "R2"]
df_all_results = pd.DataFrame(all_results, columns=columns)
df_all_results_24h = pd.DataFrame(all_results_24h, columns=columns)

print("\nRécapitulatif des métriques ELM (vs Persistence horizon):")
print(df_all_results.to_string(index=False, float_format="{:.5f}".format))

print("\nRécapitulatif des métriques Persistence 24h:")
print(df_all_results_24h.to_string(index=False, float_format="{:.5f}".format))

#---------------------------------------------------------------------------
# Create summary by horizon for both models
horizon_summary = []
horizon_summary_24h = []

for h in range(1, 25):
    # ELM metrics
    metrics = horizon_metrics[h]
    horizon_summary.append([
        h, metrics['nRMSE'], metrics['Gain'], 
        metrics['nMAE'], metrics['nMBE'], metrics['R2']
    ])
    
    # Persistence 24h metrics
    metrics_24h = horizon_metrics_24h[h]
    horizon_summary_24h.append([
        h, metrics_24h['nRMSE'], metrics_24h['Gain'], 
        metrics_24h['nMAE'], metrics_24h['nMBE'], metrics_24h['R2']
    ])

df_horizon_summary = pd.DataFrame(horizon_summary, 
                                  columns=["Horizon", "nRMSE_avg", "Gain_avg", 
                                          "nMAE_avg", "nMBE_avg", "R2_avg"])
df_horizon_summary_24h = pd.DataFrame(horizon_summary_24h, 
                                      columns=["Horizon", "nRMSE_avg", "Gain_avg", 
                                              "nMAE_avg", "nMBE_avg", "R2_avg"])

# Display results

print("RÉSULTATS ELM PAR HORIZON DE PRÉDICTION (Moyennes sur toutes les variables)")
print("="*80)
print(df_horizon_summary.to_string(index=False, float_format="{:.4f}".format))


print("RÉSULTATS PERSISTENCE 24H PAR HORIZON DE PRÉDICTION (Moyennes sur toutes les variables)")
print("="*80)
print(df_horizon_summary_24h.to_string(index=False, float_format="{:.4f}".format))

#GRAPHE 2---------------------------------------------------------------- 
# Create visualization of metrics by horizon (comparison ELM vs Persistence 24h)

fig, ((ax1, ax2)) = plt.subplots(2, figsize=(16, 12))
"""
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
"""
horizons = list(range(1, 25))

# nRMSE plot
ax1.plot(horizons, df_horizon_summary['nRMSE_avg'], 'b-o', linewidth=2, markersize=4, label='ELM')
ax1.plot(horizons, df_horizon_summary_24h['nRMSE_avg'], 'm-s', linewidth=2, markersize=4, label='Persistence 24h')
ax1.set_xlabel('Horizon de prédiction (h)')
ax1.set_ylabel('nRMSE moyen')
ax1.set_title('Évolution du nRMSE par horizon')
ax1.legend()
ax1.grid(True, alpha=0.3)

# R2 plot
ax2.plot(horizons, df_horizon_summary['R2_avg'], 'b-o', linewidth=2, markersize=4, label='ELM')
ax2.plot(horizons, df_horizon_summary_24h['R2_avg'], 'm-s', linewidth=2, markersize=4, label='Persistence 24h')
ax2.set_xlabel('Horizon de prédiction (h)')
ax2.set_ylabel('R² moyen')
ax2.set_title('Évolution du R² par horizon')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Gain plot
"""
ax3.plot(horizons, df_horizon_summary['Gain_avg'], 'b-o', linewidth=2, markersize=4, label='ELM (vs Pers. horizon)')
ax3.plot(horizons, df_horizon_summary_24h['Gain_avg'], 'm-s', linewidth=2, markersize=4, label='ELM (vs Pers. 24h)')
ax3.set_xlabel('Horizon de prédiction (h)')
ax3.set_ylabel('Gain moyen')
ax3.set_title('Évolution du Gain par horizon')
ax3.legend()
ax3.grid(True, alpha=0.3)
"""


# nMAE plot
"""
ax4.plot(horizons, df_horizon_summary['nMAE_avg'], 'b-o', linewidth=2, markersize=4, label='ELM')
ax4.plot(horizons, df_horizon_summary_24h['nMAE_avg'], 'm-s', linewidth=2, markersize=4, label='Persistence 24h')
ax4.set_xlabel('Horizon de prédiction (h)')
ax4.set_ylabel('nMAE moyen')
ax4.set_title('Évolution du nMAE par horizon')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""
#GRAPHE 3---------------------------------------------------------------------------
# Create heatmaps for detailed results by variable and horizon (ELM)

print("CRÉATION DES HEATMAPS DES RÉSULTATS DÉTAILLÉS - ELM")
print("="*80)

# Pivot tables for ELM
pivot_nrmse = df_all_results.pivot(index='Variable', columns='Horizon', values='nRMSE')
pivot_gain = df_all_results.pivot(index='Variable', columns='Horizon', values='Gain')
pivot_r2 = df_all_results.pivot(index='Variable', columns='Horizon', values='R2')

# Create heatmaps for ELM
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 16))

# nRMSE heatmap
sns.heatmap(pivot_nrmse, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'nRMSE'})
ax1.set_title('ELM - nRMSE par Variable et Horizon de Prédiction')
ax1.set_xlabel('Horizon (h)')

# Gain heatmap
sns.heatmap(pivot_gain, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2, cbar_kws={'label': 'Gain'})
ax2.set_title('ELM - Gain par Variable et Horizon de Prédiction (vs Persistence horizon)')
ax2.set_xlabel('Horizon (h)')

# R2 heatmap
sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='Blues', ax=ax3, cbar_kws={'label': 'R²'})
ax3.set_title('ELM - R² par Variable et Horizon de Prédiction')
ax3.set_xlabel('Horizon (h)')

plt.tight_layout()
plt.show()

#Affichage du temps---------------------------------------------------------------
print(f"\nTemps d'exécution total: {time.time() - start_time:.2f} secondes")