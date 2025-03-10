"""
Module: modeling.py
Description:
    This module provides functions for PLSR modeling of Raman/SERDS data.
    
    - run_pls_modeling_all: For each fluorescence (F) level, performs PLSR modeling on both the
      conventional spectra (key: "specs_1_bg_noise_mean") and SERDS difference spectra (key: "serds_bg_noise_mean").
      It computes performance metrics on test and training sets, saves CSV files if desired, and returns
      the results along with a dictionary of feature importances.
      
    - run_pls_modeling_dataset5: Handles heterogeneous Dataset 5 by combining specified FSR (fluorescence)
      level pairs. For each combination, half of the samples from each FSR level are randomly selected,
      concatenated, and then used for PLSR modeling. Performance metrics for conventional and SERDS data are computed.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression

def run_pls_modeling_all(results_dict, gen_y, initial_F_levels, test_size=0.2, random_state=42, n_components=1, csv_save_dir="Results"):
    """
    Performs PLSR modeling for each fluorescence level contained in results_dict.
    
    For each F level, this function fits a PLS regression model using both the conventional spectra
    (key: "specs_1_bg_noise_mean") and SERDS difference spectra (key: "serds_bg_noise_mean"). It then computes
    performance metrics (R², MSE, MAE, RMSEP) on both test and training splits. CSV files with the results
    are saved to csv_save_dir.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of simulated or preprocessed data keyed by fluorescence level.
    gen_y : array-like
        Target variable values.
    initial_F_levels : list
        List of fluorescence levels (keys in results_dict) to process.
    test_size : float, optional
        Fraction of data used for testing (default is 0.2).
    random_state : int, optional
        Random state for reproducibility.
    n_components : int, optional
        Number of PLSR components (default is 1).
    csv_save_dir : str, optional
        Directory to save CSV results.
    
    Returns
    -------
    plsr_results_df_base : pandas.DataFrame
        DataFrame with modeling performance metrics for each F level.
    max_argmax_rmsep_df : pandas.DataFrame
        DataFrame summarizing maximum coefficient values and their indices.
    feature_importance_dict : dict
        Dictionary with feature importance estimates for each F level.
    """
    # Prepare a base DataFrame for results
    plsr_results_df_base = pd.DataFrame(columns=[
        'F level', 'Conventional (PLSR) R^2', 'SERDS (PLSR) R^2', 
        'Conventional (PLSR) MSE', 'SERDS (PLSR) MSE', 
        'Conventional (PLSR) MAE', 'SERDS (PLSR) MAE', 
        'Conventional (PLSR) RMSEP', 'SERDS (PLSR) RMSEP',
        'Train (Conventional PLSR) R^2', 'Train (Conventional PLSR) MSE', 'Train (Conventional PLSR) MAE', 
        'Train (SERDS PLSR) R^2', 'Train (SERDS PLSR) MSE', 'Train (SERDS PLSR) MAE',
        'X_conv_train', 'X_conv_test', 'X_serds_train', 'X_serds_test', 
        'y_train', 'y_test', 'y_pred_conv_plsr', 'y_pred_serds_plsr'
    ])
    
    max_values = []
    argmax_values = []
    levels = []
    data_types = []
    rmsep_values = []
    feature_importance_dict = {}
    
    for initial_F_level in initial_F_levels:
        target_results = results_dict.get(initial_F_level)
        if target_results is not None:
            X_conv = target_results.get("specs_1_bg_noise_mean")
            X_serds = target_results.get("serds_bg_noise_mean")
            if X_conv is not None and X_serds is not None:
                X_conv_train, X_conv_test, y_train, y_test = train_test_split(
                    X_conv, gen_y, test_size=test_size, random_state=random_state
                )
                X_serds_train, X_serds_test, _, _ = train_test_split(
                    X_serds, gen_y, test_size=test_size, random_state=random_state
                )
                plsr_conv = PLSRegression(n_components=n_components)
                plsr_serds = PLSRegression(n_components=n_components)
                plsr_conv.fit(X_conv_train, y_train)
                plsr_serds.fit(X_serds_train, y_train)
                plsr_coefficients_conv = plsr_conv.coef_
                plsr_coefficients_serds = plsr_serds.coef_
                
                feature_importance_dict[initial_F_level] = {
                    'conv': np.abs(plsr_coefficients_conv).mean(axis=0),
                    'serds': np.abs(plsr_coefficients_serds).mean(axis=0)
                }
                
                max_value_conv = np.max(np.abs(plsr_coefficients_conv).mean(axis=0))
                argmax_value_conv = np.argmax(np.abs(plsr_coefficients_conv).mean(axis=0))
                max_values.append(max_value_conv)
                argmax_values.append(argmax_value_conv)
                levels.append(initial_F_level)
                data_types.append('conv')
                
                max_value_serds = np.max(np.abs(plsr_coefficients_serds).mean(axis=0))
                argmax_value_serds = np.argmax(np.abs(plsr_coefficients_serds).mean(axis=0))
                max_values.append(max_value_serds)
                argmax_values.append(argmax_value_serds)
                levels.append(initial_F_level)
                data_types.append('serds')
                
                y_pred_conv_plsr = plsr_conv.predict(X_conv_test)
                y_pred_serds_plsr = plsr_serds.predict(X_serds_test)
                y_pred_train_conv_plsr = plsr_conv.predict(X_conv_train)
                y_pred_train_serds_plsr = plsr_serds.predict(X_serds_train)
                
                r2_conv = r2_score(y_test, y_pred_conv_plsr)
                r2_serds = r2_score(y_test, y_pred_serds_plsr)
                mse_conv = mean_squared_error(y_test, y_pred_conv_plsr)
                mse_serds = mean_squared_error(y_test, y_pred_serds_plsr)
                mae_conv = mean_absolute_error(y_test, y_pred_conv_plsr)
                mae_serds = mean_absolute_error(y_test, y_pred_serds_plsr)
                rmsep_conv = np.sqrt(mse_conv)
                rmsep_serds = np.sqrt(mse_serds)
                rmsep_values.extend([rmsep_conv, rmsep_serds])
                
                train_r2_conv = r2_score(y_train, y_pred_train_conv_plsr)
                train_r2_serds = r2_score(y_train, y_pred_train_serds_plsr)
                train_mse_conv = mean_squared_error(y_train, y_pred_train_conv_plsr)
                train_mse_serds = mean_squared_error(y_train, y_pred_train_serds_plsr)
                train_mae_conv = mean_absolute_error(y_train, y_pred_train_conv_plsr)
                train_mae_serds = mean_absolute_error(y_train, y_pred_train_serds_plsr)
                
                temp_df = pd.DataFrame({
                    'F level': [initial_F_level],
                    'Conventional (PLSR) R^2': [r2_conv],
                    'SERDS (PLSR) R^2': [r2_serds],
                    'Conventional (PLSR) MSE': [mse_conv],
                    'SERDS (PLSR) MSE': [mse_serds],
                    'Conventional (PLSR) MAE': [mae_conv],
                    'SERDS (PLSR) MAE': [mae_serds],
                    'Conventional (PLSR) RMSEP': [rmsep_conv],
                    'SERDS (PLSR) RMSEP': [rmsep_serds],
                    'Train (Conventional PLSR) R^2': [train_r2_conv],
                    'Train (Conventional PLSR) MSE': [train_mse_conv],
                    'Train (Conventional PLSR) MAE': [train_mae_conv],
                    'Train (SERDS PLSR) R^2': [train_r2_serds],
                    'Train (SERDS PLSR) MSE': [train_mse_serds],
                    'Train (SERDS PLSR) MAE': [train_mae_serds],
                    'X_conv_train': [X_conv_train],
                    'X_conv_test': [X_conv_test],
                    'X_serds_train': [X_serds_train],
                    'X_serds_test': [X_serds_test],
                    'y_train': [y_train],
                    'y_test': [y_test],
                    'y_pred_conv_plsr': [y_pred_conv_plsr],
                    'y_pred_serds_plsr': [y_pred_serds_plsr]
                })
                plsr_results_df_base = pd.concat([plsr_results_df_base, temp_df], ignore_index=True)
    
    max_argmax_rmsep_df = pd.DataFrame({
        'F level': levels,
        'Data Type': data_types,
        'Max Value': max_values,
        'Argmax': argmax_values,
        'RMSEP': rmsep_values
    })
    
    os.makedirs(csv_save_dir, exist_ok=True)
    df_conv = plsr_results_df_base[['F level', 'Conventional (PLSR) R^2', 'Conventional (PLSR) MSE', 
                                     'Conventional (PLSR) MAE', 'Conventional (PLSR) RMSEP']].copy()
    df_conv.rename(columns={
        'F level': 'F (Conventional)',
        'Conventional (PLSR) R^2': '(PLSR) R^2',
        'Conventional (PLSR) MSE': '(PLSR) MSE',
        'Conventional (PLSR) MAE': '(PLSR) MAE',
        'Conventional (PLSR) RMSEP': '(PLSR) RMSEP'
    }, inplace=True)
    df_conv.to_csv(os.path.join(csv_save_dir, "PLSR_modeling_results_Conventional.csv"), index=False, float_format='%.6g')
    
    df_serds = plsr_results_df_base[['F level', 'SERDS (PLSR) R^2', 'SERDS (PLSR) MSE', 
                                      'SERDS (PLSR) MAE', 'SERDS (PLSR) RMSEP']].copy()
    df_serds.rename(columns={
        'F level': 'F (SERDS)',
        'SERDS (PLSR) R^2': '(PLSR) R^2',
        'SERDS (PLSR) MSE': '(PLSR) MSE',
        'SERDS (PLSR) MAE': '(PLSR) MAE',
        'SERDS (PLSR) RMSEP': '(PLSR) RMSEP'
    }, inplace=True)
    df_serds.to_csv(os.path.join(csv_save_dir, "PLSR_modeling_results_SERDS.csv"), index=False, float_format='%.6g')
    
    print(plsr_results_df_base[['F level', 'Conventional (PLSR) R^2', 'SERDS (PLSR) R^2', 
                                  'Conventional (PLSR) MSE', 'SERDS (PLSR) MSE', 
                                  'Conventional (PLSR) MAE', 'SERDS (PLSR) MAE', 
                                  'Conventional (PLSR) RMSEP', 'SERDS (PLSR) RMSEP',
                                  'Train (Conventional PLSR) R^2', 'Train (Conventional PLSR) MSE', 
                                  'Train (Conventional PLSR) MAE', 'Train (SERDS PLSR) R^2', 
                                  'Train (SERDS PLSR) MSE', 'Train (SERDS PLSR) MAE']])
    print(max_argmax_rmsep_df)
    
    return plsr_results_df_base, max_argmax_rmsep_df, feature_importance_dict

def run_pls_modeling_dataset5(results_dict, gen_y, 
                              combinations=None,
                              test_size=0.2, random_state=42, n_components=1,
                              csv_save_dir="Results"):
    """
    Performs PLSR modeling for Dataset 5 by combining specified FSR levels to create heterogeneous samples.

    For each FSR combination provided in the `combinations` list, this function randomly selects half
    of the samples from each F level, concatenates them, and then performs PLS regression on both the
    conventional spectra (key: "specs_1_bg_noise_mean") and the SERDS difference spectra (key: "serds_bg_noise_mean").
    
    The function calculates performance metrics (R², MSE, MAE, RMSEP) on both training and test sets,
    aggregates the results into a DataFrame, and optionally saves CSV files.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing simulated (or preprocessed) data for different fluorescence levels (FSRs).
        Each value should be a dictionary with keys:
            - "specs_1_bg_noise_mean": np.ndarray of conventional spectra (n_samples x n_points)
            - "serds_bg_noise_mean":   np.ndarray of SERDS difference spectra (n_samples x n_points)
    gen_y : array-like
        Target variable values corresponding to the samples. It is assumed these are consistent across FSR levels.
    combinations : list of tuple, optional
        List of tuples specifying FSR pairs to combine. Default is:
        [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]
    test_size : float, optional
        Fraction of the data to be used as test set (default is 0.2).
    random_state : int, optional
        Random state for reproducibility.
    n_components : int, optional
        Number of PLSR components to use (default is 1).
    csv_save_dir : str, optional
        Directory to save CSV files with modeling results. If None, no CSV is saved.

    Returns
    -------
    plsr_results_df : pandas.DataFrame
        DataFrame containing modeling performance metrics for each FSR combination.
    concatenated_data_dict : dict
        Dictionary keyed by each FSR combination tuple; each value is the concatenated data used for modeling.
    """
    # Default FSR combinations if none are provided
    if combinations is None:
        combinations = [(0, 1), (1, 10), (10, 100), (100, 1000),
                        (1000, 2000), (2000, 3000), (3000, 4000)]
    
    plsr_results_df = pd.DataFrame(columns=[
        'F level1', 'F level2',
        'Conventional (PLSR) R^2', 'SERDS (PLSR) R^2',
        'Conventional (PLSR) MSE', 'SERDS (PLSR) MSE',
        'Conventional (PLSR) MAE', 'SERDS (PLSR) MAE',
        'Conventional (PLSR) RMSEP', 'SERDS (PLSR) RMSEP',
        'Train (Conventional PLSR) R^2', 'Train (Conventional PLSR) MSE', 'Train (Conventional PLSR) MAE',
        'Train (SERDS PLSR) R^2', 'Train (SERDS PLSR) MSE', 'Train (SERDS PLSR) MAE'
    ])
    
    concatenated_data_dict = {}
    rmsep_values = []

    np.random.seed(123)
    for FSR1, FSR2 in combinations:
        # Retrieve data for each FSR level from the results_dict
        data1 = results_dict.get(FSR1)
        data2 = results_dict.get(FSR2)
        if data1 is None or data2 is None:
            continue  # skip if either level is missing

        n_samples1 = data1['serds_bg_noise_mean'].shape[0]
        n_samples2 = data2['serds_bg_noise_mean'].shape[0]
        half_n1 = n_samples1 // 2
        half_n2 = n_samples2 // 2

        indices1 = np.random.choice(n_samples1, size=half_n1, replace=False)
        indices2 = np.random.choice(n_samples2, size=half_n2, replace=False)

        # Build selected data dictionaries for each FSR level
        selected_data1 = {
            "specs_1_bg_noise_mean": data1['specs_1_bg_noise_mean'][indices1],
            "serds_bg_noise_mean": data1['serds_bg_noise_mean'][indices1],
            "y": gen_y[indices1]
        }
        selected_data2 = {
            "specs_1_bg_noise_mean": data2['specs_1_bg_noise_mean'][indices2],
            "serds_bg_noise_mean": data2['serds_bg_noise_mean'][indices2],
            "y": gen_y[indices2]
        }

        # Concatenate the data from the two FSR levels
        concatenated_data = {
            "specs_1_bg_noise_mean": np.concatenate([
                selected_data1["specs_1_bg_noise_mean"],
                selected_data2["specs_1_bg_noise_mean"]
            ], axis=0),
            "serds_bg_noise_mean": np.concatenate([
                selected_data1["serds_bg_noise_mean"],
                selected_data2["serds_bg_noise_mean"]
            ], axis=0),
            "y": np.concatenate([selected_data1["y"], selected_data2["y"]], axis=0)
        }
        concatenated_data_dict[(FSR1, FSR2)] = concatenated_data

        # Define features for conventional and SERDS spectra
        X_conv = concatenated_data["specs_1_bg_noise_mean"]
        X_serds = concatenated_data["serds_bg_noise_mean"]
        y_vals = concatenated_data["y"]

        # Split the data into train and test sets
        X_conv_train, X_conv_test, y_train, y_test = train_test_split(
            X_conv, y_vals, test_size=test_size, random_state=random_state
        )
        X_serds_train, X_serds_test, _, _ = train_test_split(
            X_serds, y_vals, test_size=test_size, random_state=random_state
        )

        # Fit PLS regression models
        plsr_conv = PLSRegression(n_components=n_components)
        plsr_serds = PLSRegression(n_components=n_components)
        plsr_conv.fit(X_conv_train, y_train)
        plsr_serds.fit(X_serds_train, y_train)

        # Predict on test and training data
        y_pred_conv = plsr_conv.predict(X_conv_test)
        y_pred_serds = plsr_serds.predict(X_serds_test)
        y_pred_train_conv = plsr_conv.predict(X_conv_train)
        y_pred_train_serds = plsr_serds.predict(X_serds_train)

        # Compute test performance metrics
        r2_conv = r2_score(y_test, y_pred_conv)
        r2_serds = r2_score(y_test, y_pred_serds)
        mse_conv = mean_squared_error(y_test, y_pred_conv)
        mse_serds = mean_squared_error(y_test, y_pred_serds)
        mae_conv = mean_absolute_error(y_test, y_pred_conv)
        mae_serds = mean_absolute_error(y_test, y_pred_serds)
        rmsep_conv = np.sqrt(mse_conv)
        rmsep_serds = np.sqrt(mse_serds)
        rmsep_values.extend([rmsep_conv, rmsep_serds])
        
        # Compute training performance metrics
        train_r2_conv = r2_score(y_train, y_pred_train_conv)
        train_r2_serds = r2_score(y_train, y_pred_train_serds)
        train_mse_conv = mean_squared_error(y_train, y_pred_train_conv)
        train_mse_serds = mean_squared_error(y_train, y_pred_train_serds)
        train_mae_conv = mean_absolute_error(y_train, y_pred_train_conv)
        train_mae_serds = mean_absolute_error(y_train, y_pred_train_serds)

        # Append the results for this combination
        temp_df = pd.DataFrame({
            'F level1': [FSR1],
            'F level2': [FSR2],
            'Conventional (PLSR) R^2': [r2_conv],
            'SERDS (PLSR) R^2': [r2_serds],
            'Conventional (PLSR) MSE': [mse_conv],
            'SERDS (PLSR) MSE': [mse_serds],
            'Conventional (PLSR) MAE': [mae_conv],
            'SERDS (PLSR) MAE': [mae_serds],
            'Conventional (PLSR) RMSEP': [rmsep_conv],
            'SERDS (PLSR) RMSEP': [rmsep_serds],
            'Train (Conventional PLSR) R^2': [train_r2_conv],
            'Train (Conventional PLSR) MSE': [train_mse_conv],
            'Train (Conventional PLSR) MAE': [train_mae_conv],
            'Train (SERDS PLSR) R^2': [train_r2_serds],
            'Train (SERDS PLSR) MSE': [train_mse_serds],
            'Train (SERDS PLSR) MAE': [train_mae_serds]
        })
        plsr_results_df = pd.concat([plsr_results_df, temp_df], ignore_index=True)

    if csv_save_dir:
        os.makedirs(csv_save_dir, exist_ok=True)
        csv_path = os.path.join(csv_save_dir, "PLSR_modeling_results_Dataset5.csv")
        plsr_results_df.to_csv(csv_path, index=False, float_format='%.6g')
    
    print(plsr_results_df)
    return plsr_results_df, concatenated_data_dict
