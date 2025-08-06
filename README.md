# Multi-Source Energy Forecasting with Optimal Reconciliation via Multi-Input Multi-Output Multi-Horizon (MIMO-MH) and Extreme Learning Machine (ELM)
## Overview
The intermittent nature of renewable energy sources poses several challenges, particularly regarding reliability, energy quality, and supply-demand balance [[1]](#ref1). In this context, forecasting electricity production from renewable sources such as wind and solar energy becomes essential for the efficient and continuous operation of the electrical grid [[2]](#ref2).

The Multi-Input Multi-Output Multi-Horizon (MIMO-MH) approach with reconciliation and Extreme Learning Machine (ELM) enables synchronization of forecasts from different sources (solar, thermal, hydraulic, imports, etc.) to ensure consistency with net consumption (equivalent to grid demand), while capturing global physical interactions and constraints (supply-demand balance, import/export, self-consumption) [[3]](#ref3).

Unlike Single-Input Single-Output (SISO) models, which process each source independently, MIMO leverages correlations between sources and shared variability, thereby improving aggregate accuracy and reducing total deviations (through the compensatory effect between errors). In parallel, ELM offers fast learning, an analytical closed-form solution, and low computational load, making it ideal for near real-time adaptation. This approach optimizes the forecasting of final demand (net consumption), which is essential for dispatching, import management, and grid stability, while providing a robust and cost-effective solution for highly variable, self-consumption-prone multi-energy systems.

## Dataset Description

The performance of energy forecasting models directly depends on the quality and representativeness of the data. To capture the dynamics of interactions between various energy sources, we use detailed hourly time series over a sufficiently long period to reflect actual variability. A time series is a sequence of observations indexed by time. In this report, the time series represent the hourly electricity production in MWh from different sources (Thermal, Hydropower, Micro-hydro, Solar PV, Wind, Bioenergy, Imports). They also include the average production cost in €/MWh and the total production in MWh. These series are managed by EDF for the Corsica region (https://opendata-corse.edf.fr/pages/home0/), covering the period from 2016 to 2022 with hourly resolution, ensuring both reliability and representativeness of the Corsican island energy context.

## Project Structure

### Core Modules

| Module | Description |
|--------|-------------|
| `config.py` | Global configuration and initial parameters |
| `data_processing.py` | Data processing and sliding window creation |
| `training.py` | ELM (Extreme Learning Machine) model training |
| `reconciliation.py` | Hierarchical Reconciliation | Forecast consistency optimization |
| `visualization.py` | Results visualization |
| `main.py` | Main module that launches functions and other modules |

### Forecasting Models

| Module | Model Type | Description |
|--------|------------|-------------|
| `persistence.py` | Naive persistence models (horizon and 24h) |
| `siso.py` | Single Input Single Output | Individual source forecasting performance |
| `mimo.py` | Multi Input Multi Output | Multi-source simultaneous forecasting |
| `mimo_mh.py` | MIMO Multi-Horizon | Extended MIMO with multiple time horizons |
| `mimo_mh_WLS.py` | MIMO-MH with WLS | Weighted Least Squares reconciliation |

## Methodology

### 1. Single Input Single Output (SISO)
Individual forecasting models for each energy source, treating them independently without considering cross-correlations.

### 2. Multi Input Multi Output (MIMO)
Simultaneous forecasting of multiple energy sources, leveraging inter-source correlations and shared variability patterns.

### 3. Multi-Horizon Extension (MIMO-MH)
Extension of MIMO to multiple forecasting horizons, enabling comprehensive temporal coverage and improved planning capabilities.

### 4. Hierarchical Forecast Reconciliation
Optimization technique ensuring forecast consistency across different aggregation levels and maintaining physical constraints (e.g., supply-demand balance).

### 5. Benchmarking Methods
- **Persistence Models**: Simple baselines using historical values
- **Advanced Baselines**: Comparison with state-of-the-art time series models

## Getting Started

### Prerequisites
- Python 3.7 or higher
- pip package manager
### Complete Installation
For a single command installation of all required packages:
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

### Usage

#### Quick Start
```bash
python main.py
```

#### Interactive Execution Pipeline

The main script provides an interactive menu-driven interface for executing various model combinations. Upon launch, users can select from predefined execution scenarios or create custom model combinations.

#### Execution Workflow

**1. Interactive Menu System**
   - Displays comprehensive model execution options
   - Supports flexible model selection through numbered choices
   - Includes custom execution mode for advanced users

**2. Automated Data Pipeline**
   - Loads energy time-series data using `load_and_preprocess_data()`
   - Preprocesses data with feature engineering and normalization
   - Creates sliding window matrices optimized for different model architectures
   - Initializes result containers for performance tracking

**3. Execution Options**

| Option | Models Executed | Description |
|--------|-----------------|-------------|
| **1** | Full Suite | SISO + MIMO + MIMO-MH + WLS + Persistence |
| **2** | Multi-Horizon Focus | MIMO-MH + MIMO-MH-WLS + Persistence |
| **3** | Advanced Models | MIMO + MIMO-MH + MIMO-MH-WLS |
| **4** | Hybrid Approach | SISO + MIMO + MIMO-MH-WLS |
| **5** | WLS Reconciliation | MIMO-MH-WLS + Persistence |
| **6** | Multi-Horizon Only | MIMO-MH + Persistence |
| **7** | MIMO Baseline | MIMO + Persistence |
| **8** | Individual Models | SISO + Persistence |
| **9** | Custom Selection | User-defined model combinations |

#### Output Generation

The pipeline automatically generates:
- Model performance metrics and comparison tables
- Forecast visualizations
- Execution logs and timing statistics

## Key Features

- **Computational Efficiency**: ELM provides fast training with analytical solutions
- **Physical Constraints**: Maintains energy balance and grid stability requirements
- **Scalability**: Handles multiple energy sources simultaneously
- **Real-time Capability**: Suitable for operational forecasting systems
- **Robustness**: Handles high variability and intermittency of renewable sources


## References
**<a id="ref1">[1]</a>** Sheraz Aslam, Herodotos Herodotou, Syed Muhammad Mohsin, Nadeem Javaid, Nouman Ashraf, and Shahzad Aslam. [A survey on deep learning methods for power load and renewable energy forecasting in smart microgrids](https://doi.org/10.1016/j.rser.2021.110992). Renewable and Sustainable Energy Reviews, 144 :110992, 2021-07-
01.

**<a id="ref2">[2]</a>** Gilles Notton, Marie-Laure Nivet, Cyril Voyant, Christophe Paoli, Christophe Darras, Fabrice Motte, and Alexis Fouilloy. [Intermittent and stochastic character of renewable energy sources : Consequences, cost of intermittence and benefit of forecasting](https://doi.org/10.1016/j.rser.2018.02.007). Renewable and Sustainable Energy Reviews, 87 :96–105, 2018.

**<a id="ref3">[3]</a>** Cyril Voyant, Milan Despotovic, Gilles Notton, Yves-Marie Saint-Drenan, Mohammed Asloune, and Luis Garcia-Gutierrez. [On the importance of clearsky model in short-term solar radiation forecasting](https://doi.org/10.1016/j.solener.2025.113490). Solar Energy, 294 :113490, 2025.
