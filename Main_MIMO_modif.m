% Read the Excel file and store the data as vectors
clear all
data = readtable('Data.xlsx');
tic

% Optimization Parameters
window_size = 48;  % Number of rows for an observation
max_horizon = 24;  % Maximum prediction horizon
numHiddenUnits = 1000;  % Number of hidden neurons
numInitializations = 1;  % Number of initializations to try

% Extract necessary columns
TotalProduction_MW = max(data{:, 4}, 0);  
Thermal_MW = max(data{:, 5}, 0);
Hydro_MW = max(data{:, 7}, 0);
Solar_MW = max(data{:, 9}, 0);
Wind_MW = max(data{:, 10}, 0);
BioEnergy_MW = max(data{:, 11}, 0);
Import_MW = max(data{:, 13}, 0);
DateTime = datetime(data{:, 3}, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ssXXX', 'TimeZone', 'Europe/Paris');
Hours = hour(DateTime);

% Transform the hour (0 to 23) with an offset
Hours_offset = Hours + 1;  % Add offset to ensure positive values
Hours_sin = sin(2 * pi * Hours_offset / 24);   
Hours_cos = cos(2 * pi * Hours_offset / 24);

% Create a matrix containing all variables
input_matrix = [TotalProduction_MW, Thermal_MW, Hydro_MW, ...
                Solar_MW, Wind_MW, BioEnergy_MW, Import_MW, ...
                Hours_sin, Hours_cos];

% Output variable names
outputNames = {'Total_MW', 'Thermal_MW', 'Hydro_MW', ...
               'Solar_MW', 'Wind_MW', 'BioEner_MW', 'Import_MW'};

% Initialize results storage
numOutputs = length(outputNames);
results_table = table();

%% Main loop over prediction horizons
for prediction_horizon = 1:max_horizon
    fprintf('Processing horizon %d/%d...\n', prediction_horizon, max_horizon);
    
    %% Input-Output with Preallocation and windowed data
    numRows = size(input_matrix, 1);
    numObservations = numRows - window_size - prediction_horizon;
    
    % Preallocate matrices X and Y
    X = zeros(numObservations, window_size * size(input_matrix, 2));  
    Y = zeros(numObservations, size(input_matrix, 2) - 2);  % Outputs except sine/cosine
    
    % Populate X and Y matrices
    for i = 1:numObservations
        X(i, :) = reshape(input_matrix(i:i+window_size-1, :)', 1, []);  
        Y(i, :) = input_matrix(i+window_size+prediction_horizon-1, 1:end-2);  
    end
    
    % Replace NaNs with 0
    X(isnan(X)) = 0;
    Y(isnan(Y)) = 0;
    
    % Prepare training and testing data
    trainSize = round(0.8 * size(X, 1));
    XTrain = X(1:trainSize, :);
    YTrain = Y(1:trainSize, :);
    XTest = X(trainSize+1:end, :);
    YTest = Y(trainSize+1:end, :);
    
    % ELM model parameters
    best_rmse = inf;  % Initialize best RMSE
    best_inputWeights = [];  % Initialize best input weights
    best_bias = [];  % Initialize best bias
    best_outputWeights = [];  % Initialize best output weights
    
    % Test multiple initializations
    for init = 1:numInitializations
        % Initialize hidden layer weights and bias
        inputWeights = rand(numHiddenUnits, size(XTrain, 2));
        bias = rand(numHiddenUnits, 1);
        
        % Compute hidden layer output
        H = max(0, XTrain * inputWeights' + bias');  % ReLU activation function
        
        % Compute output weights using linear regression
        outputWeights = pinv(H) * YTrain;  % Corrected regression
        
        % Make predictions on test set
        H_test = max(0, XTest * inputWeights' + bias');  
        YPred = H_test * outputWeights;  % Unnormalized predictions
        
        % Ensure predicted values are non-negative
        YPred = max(YPred, 0); 
        
        % Calculate error metrics
        rmse_values = zeros(numOutputs, 1);
        for j = 1:numOutputs
            y_true = YTest(:, j);
            y_pred = YPred(:, j);
            
            rmse = sqrt(mean((y_pred - y_true).^2));  % RMSE
            rmse_values(j) = rmse;  % Store RMSE for each output
        end
        
        % Compute mean RMSE across all outputs
        mean_rmse = mean(rmse_values);
        
        % Keep the best initialization
        if mean_rmse < best_rmse
            best_rmse = mean_rmse;
            best_inputWeights = inputWeights;
            best_bias = bias;
            best_outputWeights = outputWeights;
        end
    end
    
    % Final predictions with best weights
    H_final = max(0, XTest * best_inputWeights' + best_bias');  
    YPred_final = H_final * best_outputWeights;  % Unnormalized predictions
    
    % Ensure predicted values are non-negative
    YPred_final = max(YPred_final, 0); 
    
    % Final error metrics calculations
    n = size(YTest, 1);  % Number of test samples
    
    nRMSE = zeros(numOutputs, 1);
    nMAE = zeros(numOutputs, 1);
    nMBE = zeros(numOutputs, 1);
    R2 = zeros(numOutputs, 1);
    
    % Persistence model for comparison
    YPersistence = zeros(size(YTest));
    for i = 1:size(YTest, 1)
        if i - prediction_horizon > 0
            YPersistence(i, :) = YTest(i - prediction_horizon, :);
        else
            YPersistence(i, :) = 0; % Replace NaNs with 0 to handle the initial condition
        end
    end
    
    % Calculate metrics for predictions and persistence
    for j = 1:numOutputs
        y_true = YTest(:, j);
        y_pred = YPred_final(:, j);
        
        % Calculate error metrics
        rmse = sqrt(mean((y_pred - y_true).^2));  % RMSE
        mae = mean(abs(y_pred - y_true));  % MAE
        mbe = mean(y_pred - y_true);  % MBE
        
        mean_y = mean(y_true);
        nRMSE(j) = rmse / mean_y;  % Normalized RMSE
        nMAE(j) = mae / mean_y;  % Normalized MAE
        nMBE(j) = mbe / mean_y;
        
        % R² calculation
        ss_res = sum((y_true - y_pred).^2);
        ss_tot = sum((y_true - mean(y_true)).^2);
        R2(j) = 1 - (ss_res / ss_tot);
        
        % Store results in table
        new_row = table(prediction_horizon, outputNames(j), nRMSE(j), nMAE(j), nMBE(j), R2(j), ...
                       'VariableNames', {'Horizon', 'Variable', 'nRMSE', 'nMAE', 'nMBE', 'R2'});
        results_table = [results_table; new_row];
    end
    
    fprintf('Horizon %d completed\n', prediction_horizon);
end

toc

%% Display results summary
fprintf('\n=== RESULTS SUMMARY ===\n');
for h = 1:max_horizon
    fprintf('\nHorizon %d:\n', h);
    horizon_data = results_table(results_table.Horizon == h, :);
    for i = 1:height(horizon_data)
        fprintf('  %s: nRMSE=%.4f, nMAE=%.4f, nMBE=%.4f, R2=%.4f\n', ...
                horizon_data.Variable{i}, horizon_data.nRMSE(i), ...
                horizon_data.nMAE(i), horizon_data.nMBE(i), horizon_data.R2(i));
    end
end



%% Save results to Excel file
output_filename = 'ELM_Results_All_Horizons.xlsx';
writetable(results_table, output_filename);
fprintf('\nResults saved to %s\n', output_filename);

%% Create summary statistics table
summary_stats = table();
for h = 1:max_horizon
    horizon_data = results_table(results_table.Horizon == h, :);
    mean_nRMSE = mean(horizon_data.nRMSE);
    mean_nMAE = mean(horizon_data.nMAE);
    mean_nMBE = mean(horizon_data.nMBE);
    mean_R2 = mean(horizon_data.R2);
    
    new_summary = table(h, mean_nRMSE, mean_nMAE, mean_nMBE, mean_R2, ...
                       'VariableNames', {'Horizon', 'Mean_nRMSE', 'Mean_nMAE', 'Mean_nMBE', 'Mean_R2'});
    summary_stats = [summary_stats; new_summary];
end

